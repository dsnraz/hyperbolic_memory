from .hierarchical_dataset import SubtreeDataset, create_subtree_dataloader, extract_nodes_from_store
from ..stores.hierarchical_vector_store import HierarchicalVectorStore
from ..encoders import EmbeddingEncoder

PERSIST_DIR = "/share/home/leiyh5/Memory/data/hierarchical_memory2"


embedding_encoder = EmbeddingEncoder()
store = HierarchicalVectorStore(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_encoder.generate_embedding,
    delayed_write=False,
)

nodes_by_level1 = extract_nodes_from_store(store,level_pair_index=1)


# dataset = SubtreeDataset(embedding_dim=384,
#                               nodes_by_level=nodes_by_level1,
#                               device="auto",
#                               ,
#                               )

dataloador = create_subtree_dataloader(
    nodes_by_level=nodes_by_level1, 
    batch_size=1, 
    shuffle=True, 
    embedding_dim=384,
    level_pair = ("DOMAIN","CATEGORY"),
    )


count = 0

for batch in dataloador:
    parent_feats = batch.parent_feats
    child_feats = batch.child_feats
    parent_child_mask = batch.parent_child_mask
    parent_child_map = batch.parent_child_map
    print(parent_feats.shape,child_feats.shape,parent_child_mask.shape,parent_child_map.shape)
    print("-----------------------------")
    print(f"parent_feats: max={parent_feats.max().item():.6f}, min={parent_feats.min().item():.6f}, mean={parent_feats.mean().item():.6f}")
    print(f"child_feats: max={child_feats.max().item():.6f}, min={child_feats.min().item():.6f}, mean={child_feats.mean().item():.6f}")
    print("parent_child_mask:",parent_child_mask)
    print("parent_child_map:",parent_child_map)
    count += 1
    if count == 1:
        break