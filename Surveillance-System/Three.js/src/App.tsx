// ✅ src/App.tsx
import { useState } from 'react'
import SceneCanvas from './components/SceneCanvas'
import Sidebar from './components/Sidebar'
import './App.css'
import type { Vector3 } from 'three'

export default function App() {
    // ✅ 사이드바 상태 및 모델 선택 상태
    const [sidebarOpen, setSidebarOpen] = useState(false)
    const [selectedModel, setSelectedModel] = useState<{
        type: string
        id: string
        videoUrl?: string
    } | null>(null)

    // ✅ 모델 클릭 시 핸들러 (CCTV, BIG HOUSE, SPEAKER 공통)
    const handleObjectClick = (type: string, id: string) => {
        let videoUrl: string | undefined = undefined
        if (type === 'cctv') {
            videoUrl = id === 'CCTV 1' ? '/static/videos/italy.mp4' : 'http://localhost:8000/cam2'
        }
        setSelectedModel({ type, id, videoUrl })
        setSidebarOpen(true)
    }

    return (
        <div className="app">
            {/* ✅ 사이드바 영역 */}
            <Sidebar
                open={sidebarOpen}
                onClose={() => {
                    setSidebarOpen(false)
                    setSelectedModel(null)
                }}
                model={selectedModel}
            />

            {/* ✅ 3D 씬 캔버스 영역 */}
            <div className={`canvas-container${sidebarOpen ? ' shift' : ''}`}>
                <SceneCanvas
                    onObjectClick={handleObjectClick}
                    sidebarOpen={sidebarOpen}
                />
            </div>
        </div>
    )
}