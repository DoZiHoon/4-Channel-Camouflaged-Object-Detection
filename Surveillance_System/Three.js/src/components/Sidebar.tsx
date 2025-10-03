// ✅ src/components/Sidebar.tsx
import './Sidebar.css'

interface SidebarProps {
    open: boolean
    onClose: () => void
    model: { type: string; id: string; videoUrl?: string } | null
}

export default function Sidebar({ open, onClose, model }: SidebarProps) {
    if (!model) return null

    return (
        <div className={`sidebar${open ? ' open' : ''}`}>
            {/* ✅ BIG HOUSE 사이드바 */}
            {model.type === 'bighouse' && (
                <>
                    <h2>OO 소초</h2>
                    <table className="info-table">
                        <tbody>
                            <tr><td colSpan={2} className="title">OO 소초</td></tr>

                            <tr><td colSpan={2} className="section">현행 작전 부대</td></tr>
                            <tr><td colSpan={2}>12 사단</td></tr>
                            <tr><td colSpan={2}>52 사단</td></tr>
                            <tr><td colSpan={2}>2 대대</td></tr>

                            <tr><td>대대장</td><td>홍길동</td></tr>
                            <tr><td>관할 소초</td><td>O1 ~ O0</td></tr>
                            <tr><td>투입일시</td><td>2024.12.23<br/>(월)</td></tr>
                            <tr><td>철수일자</td><td>2025.06.30<br/>(월)</td></tr>
                            <tr><td>작전명</td><td>D + 186</td></tr>

                            <tr><td colSpan={2} className="section">현 소초 배치</td></tr>
                            <tr><td colSpan={2}>OO 소초</td></tr>
                            <tr><td colSpan={2}>동쪽 포대</td></tr>
                            <tr><td colSpan={2}>3 포대</td></tr>

                            <tr><td>소초장</td><td>문익</td></tr>
                            <tr><td>부소초장</td><td>홍사 xxxx</td></tr>

                            <tr><td colSpan={2} className="section">투입인원 및 현재인원</td></tr>
                            <tr><td colSpan={2}>30 + 1 / 35 + 3</td></tr>

                            <tr><td colSpan={2}>보유 장비: CCTV 4 (정찰 3, 근거리 1), TOD 1</td></tr>
                            <tr><td colSpan={2}>특이사항: 대북방송기</td></tr>
                            <tr><td colSpan={2}>출력범위: 동쪽 ~ 동쪽</td></tr>

                            <tr><td colSpan={2} className="section">인접 소초</td></tr>
                            <tr><td>O9 소초</td><td>O1 소초</td></tr>
                            <tr><td>동측: 62사단 및 공병대본부</td><td>51연대</td></tr>

                            <tr><td colSpan={2} className="section">상황실</td></tr>
                        </tbody>
                    </table>
                </>
            )}

            {/* ✅ CCTV 상세정보 및 영상 */}
            {model.type === 'cctv' && (
                <>
                    <div className="metadata-container">
                        <p>&nbsp; 제조회사: 에스원</p>
                        <p>&nbsp; 등록년월 : 2023/05/26</p>
                    </div>
                    <div className="video-container">
                        {model.videoUrl ? (
                            <img
                                src={model.videoUrl}
                                alt="MJPEG 스트림"
                                className="video-player"
                            />
                        ) : (
                            <div className="video-placeholder">영상이 없습니다</div>
                        )}
                    </div>
                </>
            )}

            {/* ✅ 스피커 정보 */}
            {model.type === 'speaker' && (
                <div className="metadata-container">
                    <p>&nbsp; 위치: 광장 중앙</p>
                    <p>&nbsp; 용도: 재난 방송/경고</p>
                    <p>&nbsp; 음향 반경: 50m</p>
                </div>
            )}

            {/* ✅ 닫기 버튼 */}
            <button className="close-btn" onClick={onClose}>
                닫기
            </button>
        </div>
    )
}