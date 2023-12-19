from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nanotron.config.lighteval_config import LightEvalConfig

@dataclass
class UploadCheckpointOnS3Args:
    """Arguments related to uploading checkpoints on s3"""

    upload_s3_path: Path
    remove_after_upload: bool
    s5cmd_numworkers: Optional[int]
    s5cmd_concurrency: Optional[int]
    s5cmd_path: Optional[Path]

    def __post_init__(self):
        if isinstance(self.upload_s3_path, str):
            self.upload_s3_path = Path(self.upload_s3_path)
        if isinstance(self.s5cmd_path, str):
            self.s5cmd_path = Path(self.s5cmd_path)


@dataclass
class CheckpointsArgs:
    """Arguments related to checkpoints:
    checkpoints_path: where to save the checkpoints
    checkpoint_interval: how often to save the checkpoints
    resume_checkpoint_path: if you want to load from a specific checkpoint path
    s3: if you want to upload the checkpoints on s3

    """

    checkpoints_path: Path
    checkpoint_interval: int
    save_initial_state: Optional[bool] = False
    resume_checkpoint_path: Optional[Path] = None
    checkpoints_path_is_shared_file_system: Optional[bool] = True
    upload: Optional[UploadCheckpointOnS3Args] = None
    eval: Optional["LightEvalConfig"] = None

    def __post_init__(self):
        if isinstance(self.checkpoints_path, str):
            self.checkpoints_path = Path(self.checkpoints_path)
        if isinstance(self.resume_checkpoint_path, str):
            self.resume_checkpoint_path = Path(self.resume_checkpoint_path)

