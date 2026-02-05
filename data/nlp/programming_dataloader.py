import os
from typing import Any, Dict, List, Optional

try:
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, load_dataset, load_from_disk
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from Datasets import Dataset as HFDataset
    from Datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset


class ProgrammingDataset(Dataset):
    """Lightweight LiveBench programming dataset wrapper with optional pre-tokenisation."""

    def __init__(self, hparams: Any):  # tokenizer lives in the collator
        if hparams.execution_mode != "pretrain":
            raise ValueError(
                "ProgrammingDataset is a pretrain dataset; no other execution modes are supported."
            )

        self.hparams = hparams
        self.max_length = hparams.context_length + 1
        self.dataset_split = getattr(hparams, "dataset_split", "test")
        self.max_examples: Optional[int] = getattr(hparams, "max_dataset_samples", None)
        self.dataset = self._load_dataset()

        if self.max_examples:
            capped = min(self.max_examples, len(self.dataset))
            if capped < len(self.dataset):
                self.dataset = self.dataset.select(range(capped))

        # Always make sure we have a single "raw_content" column before any tokenisation.
        if (
            "raw_content" not in self.dataset.column_names
            and "input_ids" not in self.dataset.column_names
        ):
            original_columns = [col for col in self.dataset.column_names if col != "raw_content"]
            self.dataset = self.dataset.map(
                self._to_raw_content_batch,
                batched=True,
                remove_columns=original_columns,
                desc="Formatting LiveBench coding samples",
            )

        self.tokenizer = None
        if self.hparams.pretokenize_dataset:
            self._ensure_tokenizer()
            num_proc = getattr(self.hparams, "dataset_map_workers", 1)
            if not num_proc or num_proc <= 1:
                num_proc = None
            padding = "max_length" if self.hparams.mcmc_replay_buffer else True
            self.dataset = self.dataset.map(
                self._tokenize_batch,
                batched=True,
                num_proc=num_proc,
                fn_kwargs={"padding": padding},
                desc="Tokenizing LiveBench coding samples",
            )
            self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        print(
            f"[ProgrammingDataset] Loaded split '{self.dataset_split}' with {len(self.dataset)} examples "
            f"(pretok={self.hparams.pretokenize_dataset}, max_examples={self.max_examples})"
        )

    # ----------------------------------------------------------------------------------
    # Dataset helpers
    # ----------------------------------------------------------------------------------
    def _load_dataset(self) -> Any:
        dataset_dir = getattr(self.hparams, "dataset_dir", "") or ""
        dataset_name = getattr(self.hparams, "dataset_name", None)
        dataset_config = getattr(self.hparams, "dataset_config", None)

        if dataset_dir:
            expanded_dir = os.path.expanduser(dataset_dir)
            if os.path.isdir(expanded_dir):
                dataset = load_from_disk(expanded_dir)
                if isinstance(dataset, DatasetDict):
                    if self.dataset_split not in dataset:
                        available = list(dataset.keys())
                        raise ValueError(
                            f"Requested split '{self.dataset_split}' not found in dataset at {expanded_dir}. "
                            f"Available splits: {available}"
                        )
                    return dataset[self.dataset_split]
                return dataset

        if not dataset_name:
            raise ValueError(
                "dataset_dir does not exist and no dataset_name was provided to download from Hugging Face."
            )

        load_kwargs: Dict[str, Any] = {}
        if dataset_dir:
            load_kwargs["cache_dir"] = dataset_dir

        dataset = load_dataset(dataset_name, dataset_config, split=self.dataset_split, **load_kwargs)
        if isinstance(dataset, DatasetDict):  # defensive in case HF returns a dict
            dataset = dataset[self.dataset_split]
        return dataset

    def _ensure_tokenizer(self) -> None:
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer, clean_up_tokenization_spaces=False
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _format_example(example: Dict[str, Any]) -> str:
        sections: List[str] = []

        title = example.get("question_title") or example.get("question_id")
        if title:
            sections.append(f"### Title: {title}")

        turns = example.get("turns") or []
        if turns:
            sections.append("\n".join(turns))

        public_tests = example.get("public_test_cases")
        if public_tests:
            sections.append(f"### Public Tests\n{public_tests}")

        solution = example.get("solution")
        if solution:
            sections.append(f"### Reference Solution\n{solution}")

        remainder = example.get("remainder")
        if remainder:
            sections.append(remainder)

        return "\n\n".join(filter(None, sections)).strip()

    def _to_raw_content_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        first_column = next(iter(batch.values())) if batch else []
        batch_size = len(first_column)
        raw: List[str] = []
        for idx in range(batch_size):
            example = {key: column[idx] for key, column in batch.items()}
            raw.append(self._format_example(example))
        return {"raw_content": raw}

    def _tokenize_batch(self, batch: Dict[str, List[str]], padding: Any = True) -> Dict[str, Any]:
        tokens = self.tokenizer(
            batch["raw_content"],
            padding=padding,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return tokens

    # ----------------------------------------------------------------------------------
    # Dataset protocol
    # ----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if self.hparams.pretokenize_dataset:
            return self.dataset[idx]
        return self.dataset[idx]["raw_content"]