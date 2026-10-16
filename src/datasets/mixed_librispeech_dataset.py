import hashlib
import json
from pathlib import Path

from src.datasets.base_dataset import BaseDataset
from src.datasets.librispeech_dataset import LibrispeechDataset


class MixedLibrispeechDataset(BaseDataset):
    """
    Deterministic mixture of several LibriSpeech training splits.

    Example:
        train-clean-100 -> all
        train-clean-360 -> 80 hours
        train-other-500 -> 120 hours

    Selection is deterministic:
    SHA256(seed + split + utterance_id)

    Therefore the same samples are selected in every Kaggle session.
    """

    MANIFEST_VERSION = 1

    def __init__(
        self,
        parts_hours,
        manifest_path,
        seed=42,
        text_encoder=None,
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        max_text_length=None,
        instance_transforms=None,
    ):
        self.parts_hours = dict(parts_hours)
        self.manifest_path = Path(manifest_path)
        self.seed = int(seed)

        # Load complete indexes using already existing
        # LibrispeechDataset implementation.
        full_indexes = {}

        for part in self.parts_hours:
            dataset = LibrispeechDataset(
                part=part,
                text_encoder=text_encoder,
                target_sr=target_sr,
                limit=None,
                max_audio_length=None,
                max_text_length=None,
                instance_transforms=instance_transforms,
            )

            full_indexes[part] = list(dataset._index)

        if self.manifest_path.exists():
            manifest = self._load_manifest()
        else:
            manifest = self._create_manifest(full_indexes)
            self._save_manifest(manifest)

        index = self._records_from_manifest(
            full_indexes,
            manifest,
        )

        self._print_statistics(index)

        super().__init__(
            index=index,
            text_encoder=text_encoder,
            target_sr=target_sr,
            limit=limit,
            max_audio_length=max_audio_length,
            max_text_length=max_text_length,
            shuffle_index=False,
            instance_transforms=instance_transforms,
        )

    @staticmethod
    def _utterance_id(record):
        return Path(record["path"]).stem

    def _stable_hash(self, part, record):
        utterance_id = self._utterance_id(record)

        value = f"{self.seed}:{part}:{utterance_id}"

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

    def _select_hours(
        self,
        part,
        records,
        target_hours,
    ):
        """
        Select approximately target_hours using stable hashing.

        audio_len in the project index is measured in seconds.
        """

        # None means take the complete split.
        if target_hours is None:
            return list(records)

        target_seconds = float(target_hours) * 3600.0

        ranked = sorted(
            records,
            key=lambda record: self._stable_hash(part, record),
        )

        selected = []
        total_seconds = 0.0

        for record in ranked:
            selected.append(record)
            total_seconds += float(record["audio_len"])

            if total_seconds >= target_seconds:
                break

        return selected

    def _create_manifest(self, full_indexes):
        manifest = {
            "version": self.MANIFEST_VERSION,
            "seed": self.seed,
            "parts_hours": self.parts_hours,
            "splits": {},
        }

        for part, target_hours in self.parts_hours.items():
            records = self._select_hours(
                part,
                full_indexes[part],
                target_hours,
            )

            ids = [
                self._utterance_id(record)
                for record in records
            ]

            actual_hours = (
                sum(float(record["audio_len"]) for record in records)
                / 3600.0
            )

            manifest["splits"][part] = {
                "utterance_ids": ids,
                "num_records": len(ids),
                "actual_hours": actual_hours,
            }

        return manifest

    def _save_manifest(self, manifest):
        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Created deterministic manifest: "
            f"{self.manifest_path}"
        )

    def _load_manifest(self):
        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        if manifest["version"] != self.MANIFEST_VERSION:
            raise RuntimeError(
                "Manifest version does not match dataset version."
            )

        if int(manifest["seed"]) != self.seed:
            raise RuntimeError(
                "Manifest seed differs from config seed."
            )

        if manifest["parts_hours"] != self.parts_hours:
            raise RuntimeError(
                "Manifest parts_hours differs from config."
            )

        return manifest

    def _records_from_manifest(
        self,
        full_indexes,
        manifest,
    ):
        result = []

        for part in self.parts_hours:
            records = full_indexes[part]

            record_by_id = {
                self._utterance_id(record): record
                for record in records
            }

            selected_ids = (
                manifest["splits"][part]["utterance_ids"]
            )

            missing = [
                utterance_id
                for utterance_id in selected_ids
                if utterance_id not in record_by_id
            ]

            if missing:
                raise RuntimeError(
                    f"{part}: {len(missing)} utterances "
                    f"from manifest are missing. "
                    f"Example: {missing[:5]}"
                )

            for utterance_id in selected_ids:
                record = dict(record_by_id[utterance_id])

                # useful for debugging/statistics
                record["source_split"] = part

                result.append(record)

        return result

    @staticmethod
    def _print_statistics(index):
        stats = {}

        for record in index:
            part = record["source_split"]

            if part not in stats:
                stats[part] = {
                    "seconds": 0.0,
                    "records": 0,
                }

            stats[part]["seconds"] += float(
                record["audio_len"]
            )
            stats[part]["records"] += 1

        print("=" * 60)
        print("Mixed LibriSpeech dataset")

        total_seconds = 0.0
        total_records = 0

        for part, values in stats.items():
            hours = values["seconds"] / 3600.0

            print(
                f"{part:20s}: "
                f"{hours:8.2f} h | "
                f"{values['records']:6d} utterances"
            )

            total_seconds += values["seconds"]
            total_records += values["records"]

        print("-" * 60)
        print(
            f"{'TOTAL':20s}: "
            f"{total_seconds / 3600.0:8.2f} h | "
            f"{total_records:6d} utterances"
        )
        print("=" * 60)