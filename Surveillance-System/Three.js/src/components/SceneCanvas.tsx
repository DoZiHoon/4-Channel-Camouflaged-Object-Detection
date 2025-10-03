// ✅ src/components/SceneCanvas.tsx 수정
import { useRef, useEffect } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import './SceneCanvas.css'

interface Props {
    onObjectClick: (type: string, id: string) => void
    sidebarOpen: boolean
}

export default function SceneCanvas({ onObjectClick, sidebarOpen }: Props) {
    const mountRef = useRef<HTMLDivElement>(null)
    const callbackRef = useRef(onObjectClick)
    callbackRef.current = onObjectClick

    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
    const controlsRef = useRef<OrbitControls | null>(null)

    useEffect(() => {
        const mount = mountRef.current!
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0x000000)

        // ✅ 카메라 & 렌더러 세팅
        const camera = new THREE.PerspectiveCamera(60, mount.clientWidth / mount.clientHeight, 0.1, 500)
        camera.position.set(0, 60, 120)
        cameraRef.current = camera

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
        renderer.setSize(mount.clientWidth, mount.clientHeight)
        mount.appendChild(renderer.domElement)

        // ✅ OrbitControls 설정
        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.enableZoom = true
        controlsRef.current = controls

        // ✅ 조명 추가
        scene.add(new THREE.AmbientLight(0xffffff, 0.5))
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
        dirLight.position.set(10, 20, 10)
        scene.add(dirLight)

        // ✅ Raycaster 준비
        const raycaster = new THREE.Raycaster()
        const pointer = new THREE.Vector2()

        const loader = new GLTFLoader()
        let islandMesh: THREE.Object3D | null = null
        let topY = 0
        const cctvItems: Array<{ mesh: THREE.Object3D; clickDisc: THREE.Mesh }> = []

        // ✅ island 모델 먼저 로딩
        loader.load('/models/island.glb', (g) => {
            islandMesh = g.scene
            islandMesh.scale.set(10, 10, 10)
            islandMesh.position.set(0, -5, 0)
            scene.add(islandMesh)
            const box = new THREE.Box3().setFromObject(islandMesh)
            topY = box.max.y

            // ✅ island 로딩 이후, JSON 기반 여러 오브젝트 추가
            fetch('/scene_data.json')
                .then((res) => res.json())
                .then((data: { objects: any[] }) => {
                    data.objects.forEach((spec, idx) => {
                        loader.load(`/models/${spec.model}.glb`, (g2) => {
                            const mesh = g2.scene
                            mesh.position.set(spec.position.x, spec.position.y, spec.position.z)
                            mesh.rotation.set(spec.rotation.x, spec.rotation.y, spec.rotation.z)
                            mesh.scale.set(spec.scale.x, spec.scale.y, spec.scale.z)
                            scene.add(mesh)

                            const clickableModels = ['cctv', 'bighouse', 'speaker']
                            if (clickableModels.includes(spec.model)) {
                                // ✅ 클릭 감지를 위한 투명한 디스크
                                const discRadius = 7
                                const segments = 64
                                const clickGeom = new THREE.CircleGeometry(discRadius, segments)
                                const clickMat = new THREE.MeshBasicMaterial({ color: 0x000000, opacity: 0, transparent: true })
                                const clickDisc = new THREE.Mesh(clickGeom, clickMat)
                                clickDisc.rotation.x = -Math.PI / 2
                                clickDisc.position.set(spec.position.x, topY + 2.5, spec.position.z)
                                clickDisc.userData.modelType = spec.model
                                clickDisc.userData.camId = `${spec.model.toUpperCase()} ${idx + 1}`
                                scene.add(clickDisc)

                                // ✅ 클릭 가능한 영역 시각화를 위한 링 추가
                                const pts: THREE.Vector3[] = []
                                for (let i = 0; i <= segments; i++) {
                                    const theta = (i / segments) * Math.PI * 2
                                    pts.push(new THREE.Vector3(Math.cos(theta) * discRadius, 0, Math.sin(theta) * discRadius))
                                }
                                const ringGeom = new THREE.BufferGeometry().setFromPoints(pts)
                                const ringMat = new THREE.LineBasicMaterial({ color: 0x87ceeb, transparent: true, opacity: 0.5 })
                                const visualRing = new THREE.LineLoop(ringGeom, ringMat)
                                visualRing.rotation.x = -Math.PI / 2
                                visualRing.position.copy(clickDisc.position)
                                scene.add(visualRing)

                                // ✅ 클릭 가능한 객체로 저장
                                cctvItems.push({ mesh, clickDisc })
                            }
                        })
                    })
                })
        })

        // ✅ 마우스 오버 시 커서 변경
        const onPointerMove = (e: MouseEvent) => {
            const rect = mount.getBoundingClientRect()
            pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
            pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
            raycaster.setFromCamera(pointer, camera)
            const intersects = cctvItems.map((it) => it.clickDisc)
            const hits = raycaster.intersectObjects(intersects, false)
            document.body.style.cursor = hits.length ? 'pointer' : 'auto'
        }
        mount.addEventListener('pointermove', onPointerMove)

        // ✅ 클릭 이벤트 핸들러
        const onClick = (e: MouseEvent) => {
            const rect = mount.getBoundingClientRect()
            pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
            pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
            raycaster.setFromCamera(pointer, camera)
            const discs = cctvItems.map((it) => it.clickDisc)
            const hits = raycaster.intersectObjects(discs, false)
            if (hits.length) {
                const disc = hits[0].object as THREE.Mesh
                const item = cctvItems.find((it) => it.clickDisc === disc)!
                console.log('✅ 클릭됨:', disc.userData) // 🔍 여기에 로그 추가
                const modelType = disc.userData.modelType as string
                const camId = disc.userData.camId as string
                callbackRef.current(modelType, camId)
                // ✅ 클릭 시 줌인
                const center = new THREE.Vector3()
                new THREE.Box3().setFromObject(item.mesh).getCenter(center)
                controlsRef.current!.target.copy(center)
                camera.position.copy(center.clone().add(new THREE.Vector3(0, 20, 40)))
                controlsRef.current!.update()
            }
        }
        mount.addEventListener('click', onClick)

        // ✅ 애니메이션 루프
        const animate = () => {
            requestAnimationFrame(animate)
            controlsRef.current!.update()
            renderer.render(scene, camera)
        }
        animate()

        // ✅ 윈도우 리사이즈 대응
        const onResize = () => {
            camera.aspect = mount.clientWidth / mount.clientHeight
            camera.updateProjectionMatrix()
            renderer.setSize(mount.clientWidth, mount.clientHeight)
        }
        window.addEventListener('resize', onResize)

        return () => {
            mount.removeEventListener('pointermove', onPointerMove)
            mount.removeEventListener('click', onClick)
            window.removeEventListener('resize', onResize)
            mount.removeChild(renderer.domElement)
        }
    }, [])

    // ✅ 사이드바 닫힐 때 카메라 초기화
    useEffect(() => {
        if (!sidebarOpen && cameraRef.current && controlsRef.current) {
            cameraRef.current.position.set(0, 60, 120)
            controlsRef.current.target.set(0, 0, 0)
            controlsRef.current.update()
        }
    }, [sidebarOpen])

    return <div className="scene-canvas" ref={mountRef} />
}
