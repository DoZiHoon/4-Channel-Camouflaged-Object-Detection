// src/utils/objectLoader.ts
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import sceneData from '../../public/scene_data.json'  // JSON ��θ� ������Ʈ ������ �°� ����

export interface ObjectSpec {
    model: string
    position: { x: number; y: number; z: number }
    rotation: { x: number; y: number; z: number }
    scale: { x: number; y: number; z: number }
}

const loader = new GLTFLoader()

export async function loadSceneData(scene: THREE.Scene): Promise<void> {
    const specs: ObjectSpec[] = sceneData.objects

    await Promise.all(
        specs.map(spec =>
            new Promise<void>(resolve => {
                loader.load(
                    `/models/${spec.model}.glb`,
                    gltf => {
                        const mesh = gltf.scene
                        mesh.position.set(
                            spec.position.x,
                            spec.position.y,
                            spec.position.z
                        )
                        mesh.rotation.set(
                            spec.rotation.x,
                            spec.rotation.y,
                            spec.rotation.z
                        )
                        mesh.scale.set(
                            spec.scale.x,
                            spec.scale.y,
                            spec.scale.z
                        )
                        scene.add(mesh)
                        resolve()
                    },
                    undefined,
                    err => {
                        console.warn(`Failed to load ${spec.model}`, err)
                        resolve()
                    }
                )
            })
        )
    )
}
