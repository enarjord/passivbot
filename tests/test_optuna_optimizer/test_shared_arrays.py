"""Tests for optuna_optimizer.shared_arrays module."""
import numpy as np
import pytest

from optuna_optimizer.shared_arrays import (
    SharedArrayAttachment,
    SharedArrayManager,
    SharedArraySpec,
    attach_shared_array,
)


class TestSharedArraySpec:
    def test_is_frozen_dataclass(self):
        spec = SharedArraySpec(name="test", shape=(10, 5), dtype="<f8")
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.name = "modified"

    def test_stores_attributes(self):
        spec = SharedArraySpec(name="test_shm", shape=(100, 20), dtype="<f4")
        assert spec.name == "test_shm"
        assert spec.shape == (100, 20)
        assert spec.dtype == "<f4"


class TestSharedArrayManager:
    def test_create_from_copies_data(self):
        manager = SharedArrayManager()
        try:
            original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            spec, view = manager.create_from(original)

            assert spec.shape == (2, 2)
            assert spec.dtype == "<f8"
            np.testing.assert_array_equal(view, original)
        finally:
            manager.cleanup()

    def test_create_from_returns_valid_spec(self):
        manager = SharedArrayManager()
        try:
            arr = np.zeros((5, 3), dtype=np.float32)
            spec, _ = manager.create_from(arr)

            assert spec.name is not None
            assert len(spec.name) > 0
            assert spec.shape == (5, 3)
            assert spec.dtype == "<f4"
        finally:
            manager.cleanup()

    def test_view_returns_same_array(self):
        manager = SharedArrayManager()
        try:
            arr = np.array([1, 2, 3], dtype=np.int64)
            spec, view1 = manager.create_from(arr)
            view2 = manager.view(spec)

            assert view1 is view2
        finally:
            manager.cleanup()

    def test_modifications_visible_through_view(self):
        manager = SharedArrayManager()
        try:
            arr = np.array([1.0, 2.0, 3.0])
            spec, view = manager.create_from(arr)

            view[0] = 999.0
            retrieved = manager.view(spec)

            assert retrieved[0] == 999.0
        finally:
            manager.cleanup()

    def test_cleanup_releases_all_memory(self):
        manager = SharedArrayManager()
        arr = np.zeros((10,))
        spec, _ = manager.create_from(arr)

        manager.cleanup()

        # After cleanup, attachment should fail
        with pytest.raises(FileNotFoundError):
            attach_shared_array(spec)

    def test_cleanup_specific_specs(self):
        manager = SharedArrayManager()
        try:
            arr1 = np.zeros((5,))
            arr2 = np.zeros((3,))
            spec1, _ = manager.create_from(arr1)
            spec2, _ = manager.create_from(arr2)

            manager.cleanup([spec1])

            # spec1 should be gone
            with pytest.raises(FileNotFoundError):
                attach_shared_array(spec1)

            # spec2 should still work
            att = attach_shared_array(spec2)
            att.close()
        finally:
            manager.cleanup()

    def test_handles_non_contiguous_arrays(self):
        manager = SharedArrayManager()
        try:
            # Create non-contiguous array via transpose
            original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
            non_contiguous = original.T  # Now it's Fortran-order
            assert not non_contiguous.flags["C_CONTIGUOUS"]

            spec, view = manager.create_from(non_contiguous)

            # Should still work - data gets copied as contiguous
            np.testing.assert_array_equal(view, non_contiguous)
        finally:
            manager.cleanup()


class TestSharedArrayAttachment:
    def test_attaches_to_shared_memory(self):
        manager = SharedArrayManager()
        try:
            arr = np.array([10.0, 20.0, 30.0])
            spec, _ = manager.create_from(arr)

            attachment = SharedArrayAttachment(spec)
            np.testing.assert_array_equal(attachment.array, arr)
            attachment.close()
        finally:
            manager.cleanup()

    def test_sees_modifications_from_manager(self):
        manager = SharedArrayManager()
        try:
            arr = np.array([1.0, 2.0])
            spec, view = manager.create_from(arr)

            attachment = SharedArrayAttachment(spec)
            view[0] = 999.0

            assert attachment.array[0] == 999.0
            attachment.close()
        finally:
            manager.cleanup()

    def test_modifications_visible_to_manager(self):
        manager = SharedArrayManager()
        try:
            arr = np.array([1.0, 2.0])
            spec, view = manager.create_from(arr)

            attachment = SharedArrayAttachment(spec)
            attachment.array[1] = 888.0

            assert view[1] == 888.0
            attachment.close()
        finally:
            manager.cleanup()

    def test_close_releases_attachment(self):
        manager = SharedArrayManager()
        try:
            arr = np.zeros((5,))
            spec, _ = manager.create_from(arr)

            attachment = SharedArrayAttachment(spec)
            attachment.close()
            # Should be able to create another attachment after closing
            attachment2 = SharedArrayAttachment(spec)
            attachment2.close()
        finally:
            manager.cleanup()


class TestAttachSharedArray:
    def test_is_convenience_function(self):
        manager = SharedArrayManager()
        try:
            arr = np.array([1, 2, 3], dtype=np.int32)
            spec, _ = manager.create_from(arr)

            attachment = attach_shared_array(spec)

            assert isinstance(attachment, SharedArrayAttachment)
            np.testing.assert_array_equal(attachment.array, arr)
            attachment.close()
        finally:
            manager.cleanup()
