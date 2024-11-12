from pathlib import Path

# Directories
package_dir = Path(__file__).parent
project_dir =   package_dir.parent
data_dir = project_dir / "data"
videos_dir = data_dir / "videos"

# Pinecone variables that are helpful
index_name = "test-vqa"