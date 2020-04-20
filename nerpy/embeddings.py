import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

# TODO: Make the precision configurable
_FLOAT_TYPE = np.float32


class WordEmbedding(ABC):
    """A mapping between strings and their vector representations.

    While this is a Mapping-like data structure, it does not actually
    implement the Python Mapping interface to avoid having to follow the
    strict interface for key, value, and item views.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of the word representations."""

    @abstractmethod
    def __contains__(self, item: str) -> bool:
        """Return whether a word has a vector associated with it.

        For implementations which can produce a vector for any string,
        this should always return True. """

    @abstractmethod
    def __getitem__(self, item: str) -> Sequence[float]:
        """Return the vector associated with a word."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of words in the vocabulary.

        For implementations that are capable of producing vectors for
        words not known at training time, it is recommended to return
        the size of the vocabulary used in training."""

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over the words in the vocabulary."""

    @abstractmethod
    def items(self) -> Iterator[Tuple[str, Sequence[float]]]:
        """Iterate over the words in the vocabulary and their associated vectors."""

    def keys(self) -> Iterator[str]:
        """Iterate over the words in the vocabulary."""
        return iter(self)


class SqliteWordEmbeddings(WordEmbedding):
    def __init__(self, path: str):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

        # Set length
        cur = self.conn.execute("SELECT COUNT(*) FROM embeddings")
        self._len = cur.fetchone()[0]
        # No coverage since this kind of output cannot even be created by this class
        if self._len == 0:
            raise ValueError("Cannot load database with no words")  # pragma: no cover

        # Set dimensionality from a single example. We're guaranteed during construction
        # that the dimensionality of all vectors in the same and that there's at least
        # one word in the vocabulary. We also check above just to be extra careful.
        _word, vec = next(self.items())
        self._dim = len(vec)

    @property
    def dim(self) -> int:
        return self._dim

    def __contains__(self, item: str) -> bool:
        result = self.conn.execute(
            "SELECT vector FROM embeddings WHERE word=?", (item,)
        ).fetchone()
        return bool(result)

    def __getitem__(self, item: str) -> Sequence[float]:
        result = self.conn.execute(
            "SELECT vector FROM embeddings WHERE word=?", (item,)
        ).fetchone()
        if result:
            return np.frombuffer(result[0], dtype=_FLOAT_TYPE)
        else:
            raise KeyError(item)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[str]:
        cur = self.conn.execute("SELECT word FROM embeddings")
        while True:
            result = cur.fetchmany()
            if not result:
                break
            for item in result:
                yield item[0]

    def items(self) -> Iterator[Tuple[str, Sequence[float]]]:
        cur = self.conn.execute("SELECT * FROM embeddings")
        while True:
            result = cur.fetchmany()
            if not result:
                break
            for item in result:
                yield item[0], np.frombuffer(item[1], dtype=_FLOAT_TYPE)

    def close(self) -> None:
        self.conn.close()

    @classmethod
    def from_text_format(
        cls,
        embeddings_path: str,
        db_path: str,
        *,
        overwrite: bool = False,
        limit: Optional[int] = None,
        batch_size: int = 100,
    ) -> "SqliteWordEmbeddings":
        if limit is not None and limit <= 0:
            raise ValueError("Limit must be a positive number")

        if os.path.exists(db_path):
            if overwrite:
                os.remove(db_path)
            else:
                raise IOError(f"DB already exists at {db_path}")

        # Using WAL doesn't make this any faster, so no pragmas set
        conn = sqlite3.connect(db_path)
        with open(embeddings_path, encoding="utf8") as embeds:
            # Get header info
            vocab_size, dim = _parse_header(next(embeds))
            if vocab_size == 0 or dim == 0:
                raise ValueError(
                    f"Cannot load empty embeddings: vocabulary {vocab_size}; dimensionality {dim}"
                )

            # Create tables
            conn.execute(
                """CREATE TABLE embeddings(
                     word TEXT PRIMARY KEY NOT NULL,
                     vector BLOB NOT NULL
                   )"""
            )

            n_loaded = 0
            batch: List[Tuple[str, bytes]] = []
            for line in embeds:
                n_loaded += 1
                splits = line.rstrip(" \n").split(" ")
                word = splits[0]
                vec = np.array(splits[1:], dtype=_FLOAT_TYPE)
                if len(vec) != dim:
                    # Note that embeddings start on the second line, so n_loaded is one less than the line num
                    raise ValueError(
                        f"Expected dimensionality {dim} word embedding on line {n_loaded + 1} "
                        f"for word {word}, found dimensionality {len(vec)}"
                    )
                batch.append((word, vec.tobytes()))
                if len(batch) == batch_size:
                    conn.executemany("INSERT INTO embeddings VALUES (?, ?)", batch)
                    batch = []

                if n_loaded == limit:
                    break

        if batch:
            conn.executemany("INSERT INTO embeddings VALUES (?, ?)", batch)
            batch = []

        # Doing one big commit at the end is fastest
        conn.commit()

        # Safety check to make sure we're creating output, since an empty table cannot be loaded
        assert n_loaded != 0, "No words loaded from input"

        # Load and perform final checks
        conn.close()
        loaded_embeddings = SqliteWordEmbeddings(db_path)
        expected_loaded = limit if limit is not None else vocab_size
        # TODO: Add error explanation about duplicate words
        assert (
            len(loaded_embeddings) == n_loaded
        ), f"Expected vocabulary of {expected_loaded}, but actual is {n_loaded}"

        return loaded_embeddings


def _parse_header(line: str) -> Tuple[int, int]:
    splits = line.split()
    if len(splits) != 2:
        raise ValueError(
            "Text format embeddings must begin with a line containing length and dimensions"
        )
    try:
        return int(splits[0]), int(splits[1])
    except ValueError:
        raise ValueError("Embeddings length and dimensionality must be integers")
