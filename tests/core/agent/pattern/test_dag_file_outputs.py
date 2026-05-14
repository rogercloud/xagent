"""
Tests for DAG plan-execute pattern file outputs functionality.
"""

from unittest.mock import Mock

import pytest

from xagent.core.agent.pattern.dag_plan_execute import DAGPlanExecutePattern
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.workspace import TaskWorkspace


@pytest.fixture
def mock_workspace_db(mocker):
    """Mock database operations for workspace to avoid DB access in tests."""

    # Mock _create_file_record to do nothing (avoid DB access)
    def mock_create_record(self, file_id, file_path, db_session=None):
        # Store file_id in cache for retrieval
        path_str = str(file_path)
        resolved_str = str(file_path.resolve())
        self._recently_registered_files[path_str] = file_id
        self._recently_registered_files[resolved_str] = file_id
        self._file_id_to_path[file_id] = file_path

    mocker.patch(
        "xagent.core.workspace.TaskWorkspace._create_file_record", mock_create_record
    )
    return mocker


class TestDAGFileOutputs:
    """Test DAG plan-execute pattern file outputs functionality."""

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_extract_file_outputs_with_workspace(self, tmp_path):
        """Test that file outputs are extracted from workspace."""
        # Create a mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create a workspace with some output files
        workspace_dir = tmp_path / "test_workspace"
        workspace = TaskWorkspace("test_task", str(workspace_dir))

        # Create some test output files
        output_file1 = workspace.output_dir / "result.txt"
        output_file1.write_text("Test result content")

        output_file2 = workspace.output_dir / "data.json"
        output_file2.write_text('{"key": "value"}')

        # Create DAG pattern with workspace
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Test file extraction
        file_outputs = dag_pattern._extract_file_outputs()

        # Should contain both files as dictionaries
        filenames = [f.get("filename", "") for f in file_outputs]
        assert "result.txt" in filenames
        assert "data.json" in filenames
        assert all(
            isinstance(f.get("file_id"), str) and f.get("file_id") for f in file_outputs
        )
        file_paths = [f.get("file_path", "") for f in file_outputs]
        assert any(str(workspace.output_dir) in path for path in file_paths)

        second_pass = dag_pattern._extract_file_outputs()
        by_name = {f["filename"]: f.get("file_id") for f in file_outputs}
        second_by_name = {f["filename"]: f.get("file_id") for f in second_pass}
        assert by_name == second_by_name

    def test_extract_file_outputs_without_workspace(self, tmp_path):
        """Test that file outputs fallback to execution results when no workspace."""
        # Create a mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create DAG pattern with workspace (required)
        workspace_dir = tmp_path / "test_workspace"
        workspace = TaskWorkspace(id="test_workspace", base_dir=str(workspace_dir))
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Test with execution results containing file references

        file_outputs = dag_pattern._extract_file_outputs()

        # Should be empty since no workspace and no fallback to execution results
        assert file_outputs == []

    def test_extract_file_outputs_empty_workspace(self, tmp_path):
        """Test that empty workspace returns no file outputs."""
        # Create a mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create an empty workspace
        workspace_dir = tmp_path / "empty_workspace"
        workspace = TaskWorkspace("empty_task", str(workspace_dir))

        # Create DAG pattern with empty workspace
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Test file extraction
        file_outputs = dag_pattern._extract_file_outputs()

        # Should be empty
        assert file_outputs == []

    def test_extract_file_outputs_from_nested_step_history(self):
        """Test that delegated tool file outputs are collected from step history."""
        mock_llm = Mock(spec=OpenAILLM)
        workspace = Mock()
        workspace.get_output_files.return_value = []
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        file_outputs = dag_pattern._extract_file_outputs_from_history(
            [
                {
                    "iteration": 1,
                    "results": [
                        {
                            "step_id": "generate_file",
                            "result": {
                                "success": True,
                                "result": {
                                    "response": "created file",
                                    "file_outputs": [
                                        {
                                            "file_id": "worker-file-id",
                                            "filename": "worker.xlsx",
                                            "file_path": "/tmp/worker.xlsx",
                                        }
                                    ],
                                },
                            },
                        }
                    ],
                }
            ]
        )

        assert file_outputs == [
            {
                "file_id": "worker-file-id",
                "filename": "worker.xlsx",
                "file_path": "/tmp/worker.xlsx",
            }
        ]

    @pytest.mark.usefixtures("mock_workspace_db")
    def test_extract_file_outputs_mixed_sources(self, tmp_path):
        """Test that workspace files take precedence over execution results."""
        # Create a mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Create a workspace with output files
        workspace_dir = tmp_path / "mixed_workspace"
        workspace = TaskWorkspace("mixed_task", str(workspace_dir))

        # Create a test output file
        workspace_file = workspace.output_dir / "workspace_result.txt"
        workspace_file.write_text("Workspace content")

        # Create DAG pattern with workspace
        dag_pattern = DAGPlanExecutePattern(llm=mock_llm, workspace=workspace)

        # Test with workspace files only (no execution results fallback)
        file_outputs = dag_pattern._extract_file_outputs()

        # Should only contain workspace files
        filenames = [f.get("filename", "") for f in file_outputs]
        assert "workspace_result.txt" in filenames
        file_paths = [f.get("file_path", "") for f in file_outputs]
        assert any(str(workspace.output_dir) in path for path in file_paths)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
