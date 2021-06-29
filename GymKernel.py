import os
import queue
import sys

import numpy as np
import pandas as pd

from message.Message import MessageType
from util.util import log_print
from Kernel import Kernel

from IPython.display import clear_output

class GymKernel(Kernel):

    # maybe decompose runner function into multiple helper functions for the use in step()



    def initRunner(self,
                   RL_agent,
                   agents=[],
                   startTime=None,
                   stopTime=None,
                   num_simulations=1,
                   defaultComputationDelay=1,
                   defaultLatency=1,
                   agentLatency=None,
                   latencyNoise=[1.0],
                   agentLatencyModel=None,
                   seed=None,
                   oracle=None,
                   log_dir=None,
                   ):
        # agents passed in is Agents object, this change enables find agent by index method
        # corresponding code related to self.agents have been changed
        self.agents = agents
        self.RL_agent = RL_agent
        self.agent_saved_states = [None] * len(self.agents)

        # placeholder for current step reward and observation
        # TODO: need to confirm observation data type
        self.current_step_observation = []
        self.current_step_reward = None

        # The kernel start and stop time (first and last timestamp in
        # the simulation, separate from anything like exchange open/close).
        self.startTime = startTime
        self.stopTime = stopTime

        # The global seed, NOT used for anything agent-related.
        self.seed = seed

        # The data oracle for this simulation, if needed.
        self.oracle = oracle

        # If a log directory was not specified, use the initial wallclock.
        self.log_dir = log_dir if log_dir else str(int(self.kernelWallClockStart.timestamp()))

        # The kernel maintains a current time for each agent to allow
        # simulation of per-agent computation delays.  The agent's time
        # is pushed forward (see below) each time it awakens, and it
        # cannot receive new messages/wakeups until the global time
        # reaches the agent's time.  (i.e. it cannot act again while
        # it is still "in the future")

        # This also nicely enforces agents being unable to act before
        # the simulation startTime.
        self.agentCurrentTimes = [self.startTime] * len(agents)

        # agentComputationDelays is in nanoseconds, starts with a default
        # value from config, and can be changed by any agent at any time
        # (for itself only).  It represents the time penalty applied to
        # an agent each time it is awakened  (wakeup or recvMsg).  The
        # penalty applies _after_ the agent acts, before it may act again.
        # this might someday change to pd.Timedelta objects.
        self.agentComputationDelays = [defaultComputationDelay] * len(agents)

        # If an agentLatencyModel is defined, it will be used instead of
        # the older, non-model-based attributes.
        self.agentLatencyModel = agentLatencyModel

        # If an agentLatencyModel is NOT defined, the older parameters:
        # agentLatency (or defaultLatency) and latencyNoise should be specified.
        # These should be considered deprecated and will be removed in the future.

        # If agentLatency is not defined, define it using the defaultLatency.
        # This matrix defines the communication delay between every pair of
        # agents.
        self.agentLatency = agentLatency if agentLatency is not None else [[defaultLatency]*len(agents)]*len(agents)

        # There is a noise model for latency, intended to be a one-sided
        # distribution with the peak at zero.  By default there is no noise
        # (100% chance to add zero ns extra delay).  Format is a list with
        # list index = ns extra delay, value = probability of this delay.
        self.latencyNoise = latencyNoise

        # The kernel maintains an accumulating additional delay parameter
        # for the current agent.  This is applied to each message sent
        # and upon return from wakeup/receiveMessage, in addition to the
        # agent's standard computation delay.  However, it never carries
        # over to future wakeup/receiveMessage calls.  It is useful for
        # staggering of sent messages.
        self.currentAgentAdditionalDelay = 0

        log_print("Kernel started: {}", self.name)
        log_print("Simulation started!")

        # Note that num_simulations has not yet been really used or tested
        # for anything.  Instead we have been running multiple simulations
        # with coarse parallelization from a shell script.

        log_print("Starting sim")

        # Event notification for kernel init (agents should not try to
        # communicate with other agents, as order is unknown).  Agents
        # should initialize any internal resources that may be needed
        # to communicate with other agents during agent.kernelStarting().
        # Kernel passes self-reference for agents to retain, so they can
        # communicate with the kernel in the future (as it does not have
        # an agentID).
        log_print("\n--- Agent.kernelInitializing() ---")
        print(self.agents)
        for agent in self.agents:
            print(agent)
            agent.kernelInitializing(self)

        # Event notification for kernel start (agents may set up
        # communications or references to other agents, as all agents
        # are guaranteed to exist now).  Agents should obtain references
        # to other agents they require for proper operation (exchanges,
        # brokers, subscription services...).  Note that we generally
        # don't (and shouldn't) permit agents to get direct references
        # to other agents (like the exchange) as they could then bypass
        # the Kernel, and therefore simulation "physics" to send messages
        # directly and instantly or to perform disallowed direct inspection
        # of the other agent's state.  Agents should instead obtain the
        # agent ID of other agents, and communicate with them only via
        # the Kernel.  Direct references to utility objects that are not
        # agents are acceptable (e.g. oracles).
        log_print("\n--- Agent.kernelStarting() ---")
        for agent in self.agents:
            print(agent)
            agent.kernelStarting(self.startTime)

        # Set the kernel to its startTime.
        self.currentTime = self.startTime
        log_print("\n--- Kernel Clock started ---")
        log_print("Kernel.currentTime is now {}", self.currentTime)

        # Start processing the Event Queue.
        log_print("\n--- Kernel Event Queue begins ---")
        log_print("Kernel will start processing messages.  Queue length: {}", len(self.messages.queue))

        # Track starting wall clock time and total message count for stats at the end.
        self.eventQueueWallClockStart = pd.Timestamp("now")
        self.ttl_messages = 0

    def stepRunner(self):
        # Process messages until there aren't any (at which point there never can
        # be again, because agents only "wake" in response to messages), or until
        # the kernel stop time is reached.

        # get dummyRL agent, currently only consider one dummyRL in one runner
        # later on, we can loop over the agents and finish operations to enable multiple dummyRL
        # RL_agent = self.agents.agent_list[self.agents.getAgentIndexByName('DummyRLExecutionAgent_name')]
        
        end_step = False
        while not end_step and not self.messages.empty() and self.currentTime and (self.currentTime <= self.stopTime):
            # Get the next message in timestamp order (delivery time) and extract it.
            self.currentTime, event = self.messages.get()
            msg_recipient, msg_type, msg = event

            clear_output(wait=True)
            print("\n", msg_type)
            print("\n", msg_recipient)
            if msg_type == MessageType.MESSAGE:
                print("\n", msg.body["msg"])

            # Periodically print the simulation time and total messages, even if muted.
            if self.ttl_messages % 100000 == 0:
                print(
                    "\n--- Simulation time: {}, messages processed: {}, wallclock elapsed: {} ---\n".format(
                        self.fmtTime(self.currentTime), self.ttl_messages, pd.Timestamp("now") - self.eventQueueWallClockStart
                    )
                )

            log_print("\n--- Kernel Event Queue pop ---")
            log_print(
                "Kernel handling {} message for agent {} at time {}",
                msg_type,
                msg_recipient,
                self.fmtTime(self.currentTime),
            )

            self.ttl_messages += 1

            # In between messages, always reset the currentAgentAdditionalDelay.
            self.currentAgentAdditionalDelay = 0

            # Dispatch message to agent.
            if msg_type == MessageType.WAKEUP:

                # Who requested this wakeup call?
                agent = msg_recipient

                # Test to see if the agent is already in the future.  If so,
                # delay the wakeup until the agent can act again.
                if self.agentCurrentTimes[agent] > self.currentTime:
                    # Push the wakeup call back into the PQ with a new time.
                    self.messages.put((self.agentCurrentTimes[agent], (msg_recipient, msg_type, msg)))
                    log_print(
                        "Agent in future: wakeup requeued for {}", self.fmtTime(self.agentCurrentTimes[agent])
                    )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agentCurrentTimes[agent] = self.currentTime

                # Wake the agent.
                self.agents[agent].wakeup(self.currentTime)

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agentCurrentTimes[agent] += pd.Timedelta(
                    self.agentComputationDelays[agent] + self.currentAgentAdditionalDelay
                )

                log_print(
                    "After wakeup return, agent {} delayed from {} to {}",
                    agent,
                    self.fmtTime(self.currentTime),
                    self.fmtTime(self.agentCurrentTimes[agent]),
                )
            
            # currently, only dummyRL agent uses CANCEL_ORDER message
            elif msg_type == MessageType.CANCEL_ORDER:
                # call agent get_reward method
                self.current_step_reward = self.RL_agent.get_reward(self.currentTime)
                # call agent cancel_order method
                self.RL_agent.cancelAllOrders(self.currentTime)


            elif msg_type == MessageType.MESSAGE:

                # Who is receiving this message?
                # ? here msg_recipient is not only the id of an agent object, but also the index in agent_list
                agent = msg_recipient

                # Test to see if the agent is already in the future.  If so,
                # delay the message until the agent can act again.
                if self.agentCurrentTimes[agent] > self.currentTime:
                    # Push the message back into the PQ with a new time.
                    self.messages.put((self.agentCurrentTimes[agent], (msg_recipient, msg_type, msg)))
                    log_print(
                        "Agent in future: message requeued for {}", self.fmtTime(self.agentCurrentTimes[agent])
                    )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agentCurrentTimes[agent] = self.currentTime

                # Deliver the message.
                self.agents[agent].receiveMessage(self.currentTime, msg)

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agentCurrentTimes[agent] += pd.Timedelta(
                    self.agentComputationDelays[agent] + self.currentAgentAdditionalDelay
                )

                log_print(
                    "After receiveMessage return, agent {} delayed from {} to {}",
                    agent,
                    self.fmtTime(self.currentTime),
                    self.fmtTime(self.agentCurrentTimes[agent]),
                )

                # TODO: intercept dummy RL's QUERY SPREAD message, call get_observation, and change end_step
                # match msg_recipient id with dummy rl agent, and match msg title
                if agent == self.RL_agent.id and msg.body["msg"] == "QUERY_SPREAD":
                    self.current_step_observation = self.RL_agent.get_observation(self.currentTime)
                    end_step = True

            else:
                raise ValueError(
                    "Unknown message type found in queue",
                    "currentTime:",
                    self.currentTime,
                    "messageType:",
                    self.msg.type,
                )

            if msg_recipient == 2:
                break

        if self.messages.empty() or (self.currentTime and (self.currentTime > self.stopTime)):
            self.terminateRunner()
        
        return self.current_step_reward, self.current_step_observation


    def terminateRunner(self):
        if self.messages.empty():
            log_print("\n--- Kernel Event Queue empty ---")

        if self.currentTime and (self.currentTime > self.stopTime):
            log_print("\n--- Kernel Stop Time surpassed ---")

        # Record wall clock stop time and elapsed time for stats at the end.
        eventQueueWallClockStop = pd.Timestamp("now")

        eventQueueWallClockElapsed = eventQueueWallClockStop - self.eventQueueWallClockStart

        # Event notification for kernel end (agents may communicate with
        # other agents, as all agents are still guaranteed to exist).
        # Agents should not destroy resources they may need to respond
        # to final communications from other agents.
        log_print("\n--- Agent.kernelStopping() ---")
        for agent in self.agents:
            agent.kernelStopping()

        # Event notification for kernel termination (agents should not
        # attempt communication with other agents, as order of termination
        # is unknown).  Agents should clean up all used resources as the
        # simulation program may not actually terminate if num_simulations > 1.
        log_print("\n--- Agent.kernelTerminating() ---")
        for agent in self.agents:
            agent.kernelTerminating()

        print(
            "Event Queue elapsed: {}, messages: {}, messages per second: {:0.1f}".format(
                eventQueueWallClockElapsed,
                self.ttl_messages,
                self.ttl_messages / (eventQueueWallClockElapsed / (np.timedelta64(1, "s"))),
            )
        )
        log_print("Ending sim")

        # Agents will request the Kernel to serialize their agent logs, usually
        # during kernelTerminating, but the Kernel must write out the summary
        # log itself.
        self.writeSummaryLog()

        # This should perhaps be elsewhere, as it is explicitly financial, but it
        # is convenient to have a quick summary of the results for now.
        print("Mean ending value by agent type:")
        for a in self.meanResultByAgentType:
            value = self.meanResultByAgentType[a]
            count = self.agentCountByType[a]
            print("{}: {:d}".format(a, int(round(value / count))))

        print("Simulation ending!")

        return self.agent_saved_states

    
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