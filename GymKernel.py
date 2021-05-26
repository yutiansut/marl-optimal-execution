import os
import queue
import sys

import numpy as np
import pandas as pd

from message.Message import MessageType
from util.util import log_print
from Kernel import Kernel

class GymKernel(Kernel):

	# maybe decompose runner function into multiple helper functions for the use in step()

	def setCancelOrder(self, sender=None, requestedTime=None):
		"""
		Put CancelOrder msg into msg queue based on requestedTiem
		"""
		if requestedTime is None:
			# default is for TimeDelta is 'ns'
			# We want CancelOrder signal arrive just before next wakeup signal in the queue
			requestedTime = self.currentTime + pd.TimeDelta(0.5) # wakeup is 1 ns

		if sender is None:
			raise ValueError(
				"setCancelOrder() called without valid sender ID", "sender:", sender, "requestedTime:", requestedTime
			)

		if self.currentTime and (requestedTime < self.currentTime):
			raise ValueError(
				"setCancelOrder() called with requested time not in future",
				"currentTime:",
				self.currentTime,
				"requestedTime:",
				requestedTime,
			)

		log_print("Kernel adding CancelOrder for agent {} at time {}", sender, self.fmtTime(requestedTime))

		self.messages.put((requestedTime, (sender, MessageType.CANCEL_ORDER, None)))