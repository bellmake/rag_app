from abc import ABC, abstractmethod

class BaseChain(ABC):
    """
    모든 체인 클래스의 추상 기반 클래스.
    Args:
        model (str): 사용할 LLM 모델명
        temperature (float): LLM temperature 값
    """
    def __init__(self, model: str = "exaone", temperature: float = 0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        """체인 설정 및 반환 (구현 필수)"""
        pass

    def create(self):
        """체인 인스턴스 생성 및 반환"""
        return self.setup()
