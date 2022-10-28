"""Observation types mainly define the way data samples are observed:
   batch(multiple epochs) vs. online(one epoch)

"""
from .batch_observation import BatchObservation
from .online_observation import OnlineObservation
