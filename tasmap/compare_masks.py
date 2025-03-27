import os
from PIL import Image

# 디렉토리 경로
mask1_dir = '/workspace/MaskClustering/data/tasmap/processed/scene0000_00/output/vis_mask'
mask2_dir = '/workspace/MaskClustering/data/tasmap/processed/scene0000_00/output_crop/vis_mask'
output_dir = '/workspace/MaskClustering/data/tasmap/processed/scene0000_00/output_combined/vis_mask'

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 파일 리스트 (공통 파일명 기준 정렬)
filenames = sorted(set(os.listdir(mask1_dir)) & set(os.listdir(mask2_dir)))

# 검은 선 두께 (픽셀)
separator_height = 2

for filename in filenames:
    path1 = os.path.join(mask1_dir, filename)
    path2 = os.path.join(mask2_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # 이미지 열기
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    # 공통 너비 계산
    new_width = max(img1.width, img2.width)
    new_height = img1.height + separator_height + img2.height

    # 새 이미지 생성
    new_img = Image.new('RGB', (new_width, new_height), (0, 0, 0))  # 전체 배경을 검정으로

    # 이미지 붙이기
    new_img.paste(img1, (0, 0))
    # separator는 배경이 이미 검정이므로 따로 붙일 필요 없음
    new_img.paste(img2, (0, img1.height + separator_height))

    # 저장
    new_img.save(output_path)

print(f"검정 선 포함한 이미지가 '{output_dir}'에 저장되었습니다.")
