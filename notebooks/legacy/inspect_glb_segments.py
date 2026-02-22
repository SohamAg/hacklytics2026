from pathlib import Path
import sys

try:
    from pygltflib import GLTF2
except Exception as e:
    print('ERROR: pygltflib not installed in this environment.')
    print('Install with: pip install pygltflib')
    raise


def main():
    if len(sys.argv) < 2:
        print('Usage: python notebooks/inspect_glb_segments.py heart_models/cardiac_conduction_system.glb')
        sys.exit(1)

    glb_path = Path(sys.argv[1])
    if not glb_path.exists():
        print(f'Missing: {glb_path}')
        sys.exit(1)

    gltf = GLTF2().load(str(glb_path))

    mesh_count = len(gltf.meshes) if gltf.meshes else 0
    node_count = len(gltf.nodes) if gltf.nodes else 0
    mat_count = len(gltf.materials) if gltf.materials else 0

    print('GLB:', glb_path)
    print('Meshes:', mesh_count)
    print('Nodes:', node_count)
    print('Materials:', mat_count)

    if gltf.meshes:
        print('\nMesh names (first 50):')
        for m in gltf.meshes[:50]:
            print('-', m.name)

    if gltf.nodes:
        print('\nNode names (first 50):')
        for n in gltf.nodes[:50]:
            print('-', n.name)

    # Map nodes to meshes if possible
    if gltf.nodes and gltf.meshes:
        print('\nNodes with mesh indices (first 50):')
        shown = 0
        for i, n in enumerate(gltf.nodes):
            if n.mesh is not None:
                mesh_name = gltf.meshes[n.mesh].name if gltf.meshes[n.mesh] else None
                print(f'- node[{i}] name={n.name} -> mesh[{n.mesh}] name={mesh_name}')
                shown += 1
                if shown >= 50:
                    break

if __name__ == '__main__':
    main()
