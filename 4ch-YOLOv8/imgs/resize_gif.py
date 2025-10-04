from PIL import Image, ImageSequence

def compress_gif_pillow(input_path, output_path, resize_ratio=0.5, optimize=True):
    """
    Pillow로 GIF 용량을 줄이되 재생 속도(fps)는 유지하는 함수.
    
    :param input_path: 원본 GIF 경로
    :param output_path: 압축된 GIF 저장 경로
    :param resize_ratio: 크기 축소 비율 (ex. 0.5는 50% 크기)
    :param optimize: 색상 최적화 여부
    """
    with Image.open(input_path) as im:
        frames = []
        duration = im.info.get('duration', 100)  # 프레임 간 간격(ms)

        for frame in ImageSequence.Iterator(im):
            resized_frame = frame.resize(
                (int(frame.width * resize_ratio), int(frame.height * resize_ratio)),
                resample=Image.LANCZOS
            )
            frames.append(resized_frame.convert("P", palette=Image.ADAPTIVE))

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration,
            optimize=optimize
        )

    print(f"✅ GIF 압축 완료: {output_path} (재생 속도 유지)")

# 예시 실행
compress_gif_pillow("image107.gif", "107compressed.gif", resize_ratio=0.4)