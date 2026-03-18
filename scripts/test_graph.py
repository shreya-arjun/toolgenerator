from pathlib import Path
from toolgenerator.registry import ToolRegistry
from toolgenerator.graph import build_tool_graph, write_tool_graph, read_tool_graph, ToolGraphSampler

print("Loading registry...")
reg = ToolRegistry.from_toolbench_path(Path('data/toolenv/tools'))
print(f"Registry loaded: {len(reg.list_tools())} tools")

print("Building graph (this may take 1-2 mins)...")
G = build_tool_graph(reg)
print(f'Nodes: {G.number_of_nodes()}')
print(f'Edges: {G.number_of_edges()}')

print("Testing serialization...")
write_tool_graph(G, 'artifacts/tool_graph.gpickle')
G2 = read_tool_graph('artifacts/tool_graph.gpickle')
print(f'Round-trip OK: {G2.number_of_nodes()} nodes')

print("Testing multi-step sampler...")
sampler = ToolGraphSampler(G, reg, seed=42)
chain = sampler.sample_multi_step_chain(length=3)
print(f'Multi-step chain: {[e.endpoint_id for e in chain]}')

print("Testing parallel sampler...")
parallel = sampler.sample_parallel(count=3)
print(f'Parallel endpoints: {[e.endpoint_id for e in parallel]}')

chain2 = sampler.sample_multi_step_chain(length=3)
print(f'Second chain different: {chain[0].endpoint_id != chain2[0].endpoint_id}')

print("All checks done!")