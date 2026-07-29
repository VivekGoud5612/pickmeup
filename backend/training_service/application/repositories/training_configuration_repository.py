"""
Configuration gets its own repo
"""

from __future__ import annotations 
from abc import ABC 
from uuid import UUID 

from training_service.domain.entities.training_config import TrainingConfiguration

class TrainingConfigurationRepository(ABC):

    @abstractmethod
    def save(
        self,
        configuration: TrainingConfiguration,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        configuration: TrainingConfiguration,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(
        self,
        configuration_id: UUID,
    ) -> TrainingConfiguration:
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> list[TrainingConfiguration]:
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        configuration_id: UUID,
    ) -> None:
        raise NotImplementedError

    
    @abstractmethod
    def list_versions(self, config_name : str) -> list[TrainingConfiguration]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_version(self, config_name : str) -> TrainingConfiguration:
        raise NotImplementedError