"""Screenshot watchdog for handling screenshot requests using CDP."""

import asyncio
import os
from typing import TYPE_CHECKING, Any, ClassVar

from bubus import BaseEvent
from cdp_use.cdp.page import CaptureScreenshotParameters

from browser_use.browser.events import ScreenshotEvent
from browser_use.browser.views import BrowserError
from browser_use.browser.watchdog_base import BaseWatchdog
from browser_use.observability import observe_debug

if TYPE_CHECKING:
	pass


class ScreenshotWatchdog(BaseWatchdog):
	"""Handles screenshot requests using CDP."""

	DEFAULT_CAPTURE_COMMAND_TIMEOUT_SECONDS: ClassVar[float] = 10.0
	CAPTURE_COMMAND_TIMEOUT_ENV: ClassVar[str] = 'BROWSER_USE_SCREENSHOT_COMMAND_TIMEOUT_SECONDS'
	MAX_CAPTURE_RECOVERY_RETRIES: ClassVar[int] = 1

	# Events this watchdog listens to
	LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [ScreenshotEvent]

	# Events this watchdog emits
	EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

	def _resolve_capture_command_timeout_seconds(self, event: ScreenshotEvent) -> float:
		configured_timeout = self.DEFAULT_CAPTURE_COMMAND_TIMEOUT_SECONDS
		env_value = os.environ.get(self.CAPTURE_COMMAND_TIMEOUT_ENV)
		if env_value:
			try:
				parsed = float(env_value)
				if parsed > 0:
					configured_timeout = parsed
			except ValueError:
				self.logger.debug(
					f'[ScreenshotWatchdog] Ignoring invalid {self.CAPTURE_COMMAND_TIMEOUT_ENV}={env_value!r}'
				)

		event_timeout = float(event.event_timeout) if event.event_timeout is not None else None
		if event_timeout is None:
			return configured_timeout

		# Leave headroom for reconnect/retry and for the event bus bookkeeping timeout.
		return max(0.01, min(configured_timeout, max(event_timeout - 2.0, 0.01)))

	def _is_retryable_capture_error(self, exc: Exception) -> bool:
		if isinstance(exc, TimeoutError):
			return True

		error_text = str(exc).lower()
		retryable_markers = (
			'client is not started',
			'connection closed',
			'not connected',
			'session with given id not found',
			'target closed',
		)
		return any(marker in error_text for marker in retryable_markers)

	def _resolve_screenshot_target_id(self) -> str:
		focused_target = self.browser_session.get_focused_target()

		if focused_target and focused_target.target_type in ('page', 'tab'):
			return focused_target.target_id

		target_type_str = focused_target.target_type if focused_target else 'None'
		self.logger.warning(f'[ScreenshotWatchdog] Focused target is {target_type_str}, falling back to page target')
		page_targets = self.browser_session.get_page_targets()
		if not page_targets:
			raise BrowserError('[ScreenshotWatchdog] No page targets available for screenshot')
		return page_targets[-1].target_id

	@staticmethod
	def _build_capture_params(event: ScreenshotEvent) -> CaptureScreenshotParameters:
		params_dict: dict[str, Any] = {'format': 'png', 'captureBeyondViewport': event.full_page}
		if event.clip:
			params_dict['clip'] = {
				'x': event.clip['x'],
				'y': event.clip['y'],
				'width': event.clip['width'],
				'height': event.clip['height'],
				'scale': 1,
			}
		return CaptureScreenshotParameters(**params_dict)

	async def _capture_screenshot_once(
		self,
		*,
		target_id: str,
		params: CaptureScreenshotParameters,
		timeout_seconds: float,
	) -> str:
		cdp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=True)
		result = await asyncio.wait_for(
			cdp_session.cdp_client.send.Page.captureScreenshot(
				params=params,
				session_id=cdp_session.session_id,
			),
			timeout=timeout_seconds,
		)

		if result and 'data' in result:
			self.logger.debug('[ScreenshotWatchdog] Screenshot captured successfully')
			return result['data']

		raise BrowserError('[ScreenshotWatchdog] Screenshot result missing data')

	async def _recover_after_capture_failure(self, target_id: str, exc: Exception) -> None:
		self.logger.warning(
			f'[ScreenshotWatchdog] Screenshot capture failed for target {target_id[:8]}... '
			f'with {type(exc).__name__}: {exc}. Reconnecting CDP and retrying once.'
		)
		await self.browser_session.reconnect()
		await self.browser_session.get_or_create_cdp_session(target_id, focus=True)

	@observe_debug(ignore_input=True, ignore_output=True, name='screenshot_event_handler')
	async def on_ScreenshotEvent(self, event: ScreenshotEvent) -> str:
		"""Handle screenshot request using CDP.

		Args:
			event: ScreenshotEvent with optional full_page and clip parameters

		Returns:
			Dict with 'screenshot' key containing base64-encoded screenshot or None
		"""
		self.logger.debug('[ScreenshotWatchdog] Handler START - on_ScreenshotEvent called')
		try:
			# Remove highlights BEFORE taking the screenshot so they don't appear in the image.
			# Done here (not in finally) so CancelledError is never swallowed — any await in a
			# finally block can suppress external task cancellation.
			# remove_highlights() has its own asyncio.timeout(3.0) internally so it won't block.
			try:
				await self.browser_session.remove_highlights()
			except Exception:
				pass

			target_id = self._resolve_screenshot_target_id()
			params = self._build_capture_params(event)
			timeout_seconds = self._resolve_capture_command_timeout_seconds(event)

			self.logger.debug(f'[ScreenshotWatchdog] Taking screenshot with params: {params}')
			last_error: Exception | None = None
			for attempt in range(self.MAX_CAPTURE_RECOVERY_RETRIES + 1):
				try:
					return await self._capture_screenshot_once(
						target_id=target_id,
						params=params,
						timeout_seconds=timeout_seconds,
					)
				except Exception as exc:
					last_error = exc
					if attempt >= self.MAX_CAPTURE_RECOVERY_RETRIES or not self._is_retryable_capture_error(exc):
						break
					await self._recover_after_capture_failure(target_id, exc)

			if isinstance(last_error, TimeoutError):
				raise BrowserError(
					f'[ScreenshotWatchdog] Screenshot capture timed out after {timeout_seconds:.1f}s'
				) from last_error
			if last_error is not None:
				raise last_error
			raise BrowserError('[ScreenshotWatchdog] Screenshot failed without a captured error')
		except Exception as e:
			self.logger.error(f'[ScreenshotWatchdog] Screenshot failed: {e}')
			raise
