from abc import ABC, abstractmethod
from typing import Callable, Generator, Tuple, Literal
import cv2
import numpy as np
from pydantic import BaseModel


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float


VideoMode = Literal["RGB", "BGR"]
FrameGenerator = Generator[Tuple[np.ndarray, int], None, None]


class Video(ABC):
    def __init__(self, metadata: VideoMetadata, mode: VideoMode = "RGB"):
        self.metadata = metadata
        self.mode = mode

    @abstractmethod
    def __call__(self) -> FrameGenerator:
        pass

    def save(self, filepath: str) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            filepath,
            fourcc,
            self.metadata.fps,
            (self.metadata.width, self.metadata.height),
        )

        for frame, _ in self():
            if self.mode == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()


class FileVideo(Video):
    def __init__(
        self, metadata: VideoMetadata, load_filepath: str, mode: VideoMode = "RGB"
    ):
        super().__init__(metadata, mode)
        self.load_filepath = load_filepath
        self.cap = cv2.VideoCapture(self.load_filepath)

    def __call__(self) -> FrameGenerator:
        frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.mode == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame, frame_index
            frame_index += 1

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()


class GeneratorVideo(Video):
    def __init__(
        self,
        metadata: VideoMetadata,
        generator: FrameGenerator,
        mode: VideoMode = "RGB",
    ):
        super().__init__(metadata, mode)
        self.generator = generator

    def __call__(self) -> FrameGenerator:
        return self.generator
