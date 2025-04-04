import os
import numpy as np
import cv2
from tqdm import tqdm
from utils.config import get_args, get_dataset
import open3d as o3d
import matplotlib.pyplot as plt

def mask2box(mask):
    pos = np.where(mask)
    top = np.min(pos[0])
    bottom = np.max(pos[0])
    left = np.min(pos[1])
    right = np.max(pos[1])
    return left, top, right, bottom

def draw_bbox_on_image(rgb_image, mask):
    mask = cv2.resize(mask.astype(np.uint8), (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    left, top, right, bottom = mask2box(mask)
    output = rgb_image.copy()
    cv2.rectangle(output, (left, top), (right, bottom), (255, 0, 0), 2)  # 빨간 박스
    return output


def project(pcd, intrinsic_cam_parameters, extrinsics):
    """
    pcd: open3d.geometry.PointCloud
        - 3차원 점들의 위치 pcd.points (Nx3), 색상 pcd.colors (Nx3) 정보를 가짐
    intrinsic_cam_parameters: open3d.camera.PinholeCameraIntrinsic
        - 카메라 내부 파라미터 (fx, fy, cx, cy, 이미지 크기 등)
    extrinsics: np.ndarray, shape=(4,4)
        - "카메라→월드" 변환행렬 (Camera to World)
          따라서, 월드→카메라 변환행렬은 이 행렬의 역행렬로 구함
    return:
        - 투영된 컬러 이미지 (H x W x 3, dtype=np.uint8)
    """

    # --- 1) PinholeCameraIntrinsic 에서 내부파라미터 획득 ---
    fx = intrinsic_cam_parameters.intrinsic_matrix[0, 0]
    fy = intrinsic_cam_parameters.intrinsic_matrix[1, 1]
    cx = intrinsic_cam_parameters.intrinsic_matrix[0, 2]
    cy = intrinsic_cam_parameters.intrinsic_matrix[1, 2]

    width = intrinsic_cam_parameters.width
    height = intrinsic_cam_parameters.height

    # --- 2) pcd에서 좌표/색상 정보 추출 ---
    points = np.asarray(pcd.points)    # (N, 3)
    colors = np.asarray(pcd.colors)    # (N, 3), 범위 [0,1]

    # --- 3) extrinsics가 "카메라 -> 월드" 이므로, 월드 -> 카메라로 쓰려면 inverse 필요 ---
    world_to_cam = np.linalg.inv(extrinsics)  # (4,4)

    # 동차좌표(homogeneous)로 확장
    ones = np.ones((len(points), 1), dtype=np.float32)
    points_hom = np.hstack([points, ones])  # (N, 4)

    # --- 4) 월드 좌표계의 점들을 카메라 좌표계로 변환 ---
    camera_points = (world_to_cam @ points_hom.T).T  # (N, 4)

    x_cam = camera_points[:, 0]
    y_cam = camera_points[:, 1]
    z_cam = camera_points[:, 2]

    # --- 5) Pinhole 모델로 투영 (u, v) = (fx*x/z + cx, fy*y/z + cy) ---
    # 일반적으로 z>0 영역(카메라 앞)만 표시 가능
    u = fx * (x_cam / z_cam) + cx
    v = fy * (y_cam / z_cam) + cy

    # --- 6) Z-buffer와 결과 이미지를 준비 ---
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)

    # --- 7) 각 점을 (u,v) 위치에 찍으면서 Z-buffer 갱신 ---
    for i in range(len(points)):
        if z_cam[i] <= 0:
            # 카메라 뒤쪽(z<=0)은 투영되지 않음
            continue

        px = int(round(u[i]))
        py = int(round(v[i]))

        if 0 <= px < width and 0 <= py < height:
            if z_cam[i] < z_buffer[py, px]:
                z_buffer[py, px] = z_cam[i]
                color_int = (colors[i] * 255).astype(np.uint8)
                image[py, px] = color_int

    return image


def project(pcd, intrinsic_cam_parameters, extrinsics):
    """
    pcd: open3d.geometry.PointCloud
        - 3차원 점들의 위치 pcd.points (Nx3), 색상 pcd.colors (Nx3) 정보를 가짐
    intrinsic_cam_parameters: open3d.camera.PinholeCameraIntrinsic
        - 카메라 내부 파라미터 (fx, fy, cx, cy, 이미지 크기 등)
    extrinsics: np.ndarray, shape=(4,4)
        - "카메라→월드" 변환행렬 (Camera to World)
          따라서, 월드→카메라 변환행렬은 이 행렬의 역행렬로 구함
    return:
        - 투영된 컬러 이미지 (H x W x 3, dtype=np.uint8)
    """

    # --- 1) PinholeCameraIntrinsic 에서 내부파라미터 획득 ---
    fx = intrinsic_cam_parameters.intrinsic_matrix[0, 0]
    fy = intrinsic_cam_parameters.intrinsic_matrix[1, 1]
    cx = intrinsic_cam_parameters.intrinsic_matrix[0, 2]
    cy = intrinsic_cam_parameters.intrinsic_matrix[1, 2]

    width = intrinsic_cam_parameters.width
    height = intrinsic_cam_parameters.height

    # --- 2) pcd에서 좌표/색상 정보 추출 ---
    points = np.asarray(pcd.points)    # (N, 3)
    colors = np.asarray(pcd.colors)    # (N, 3), 범위 [0,1]

    # --- 3) extrinsics가 "카메라 -> 월드" 이므로, 월드 -> 카메라로 쓰려면 inverse 필요 ---
    world_to_cam = np.linalg.inv(extrinsics)  # (4,4)

    # 동차좌표(homogeneous)로 확장
    ones = np.ones((len(points), 1), dtype=np.float32)
    points_hom = np.hstack([points, ones])  # (N, 4)

    # --- 4) 월드 좌표계의 점들을 카메라 좌표계로 변환 ---
    camera_points = (world_to_cam @ points_hom.T).T  # (N, 4)

    x_cam = camera_points[:, 0]
    y_cam = camera_points[:, 1]
    z_cam = camera_points[:, 2]

    # --- 5) Pinhole 모델로 투영 (u, v) = (fx*x/z + cx, fy*y/z + cy) ---
    # 일반적으로 z>0 영역(카메라 앞)만 표시 가능
    u = fx * (x_cam / z_cam) + cx
    v = fy * (y_cam / z_cam) + cy

    # --- 6) Z-buffer와 결과 이미지를 준비 ---
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)


    # bounding box 추적용 (초기값 설정)
    px_min, py_min = width, height
    px_max, py_max = 0, 0

    # --- 7) 각 점을 (u,v) 위치에 찍으면서 Z-buffer 갱신 ---
    for i in range(len(points)):
        # 카메라 뒤쪽(z<=0)은 투영되지 않음
        if z_cam[i] <= 0:
            continue

        px = int(round(u[i]))
        py = int(round(v[i]))

        if 0 <= px < width and 0 <= py < height:
            if z_cam[i] < z_buffer[py, px]:
                z_buffer[py, px] = z_cam[i]
                color_int = (colors[i] * 255).astype(np.uint8)
                image[py, px] = color_int

                # bounding box 갱신
                if px < px_min:
                    px_min = px
                if px > px_max:
                    px_max = px
                if py < py_min:
                    py_min = py
                if py > py_max:
                    py_max = py

    # 유효 범위를 가지는 bbox가 없으면 None 처리
    if px_max < px_min or py_max < py_min:
        bbox = None
    else:
        bbox = (px_min, py_min, px_max, py_max)

    return image, bbox


def get_bbox_by_projection(pcd, intrinsic_cam_parameters, extrinsics):
    """
    pcd: open3d.geometry.PointCloud
        - 3차원 점들의 위치 pcd.points (Nx3), 색상 pcd.colors (Nx3) 정보를 가짐
    intrinsic_cam_parameters: open3d.camera.PinholeCameraIntrinsic
        - 카메라 내부 파라미터 (fx, fy, cx, cy, 이미지 크기 등)
    extrinsics: np.ndarray, shape=(4,4)
        - "카메라→월드" 변환행렬 (Camera to World)
          따라서, 월드→카메라 변환행렬은 이 행렬의 역행렬로 구함
    return:
        - 투영된 2D bounding box (px_min, py_min, px_max, py_max) 또는 None
    """

    # 내부 파라미터 추출
    fx = intrinsic_cam_parameters.intrinsic_matrix[0, 0]
    fy = intrinsic_cam_parameters.intrinsic_matrix[1, 1]
    cx = intrinsic_cam_parameters.intrinsic_matrix[0, 2]
    cy = intrinsic_cam_parameters.intrinsic_matrix[1, 2]
    width = intrinsic_cam_parameters.width
    height = intrinsic_cam_parameters.height

    # 점 추출
    points = np.asarray(pcd.points)

    # 카메라 좌표계로 변환
    world_to_cam = np.linalg.inv(extrinsics)
    points_hom = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    camera_points = (world_to_cam @ points_hom.T).T

    x_cam, y_cam, z_cam = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]

    # Z가 양수인 (카메라 앞에 있는) 점들만 처리
    valid_mask = z_cam > 0
    x_cam, y_cam, z_cam = x_cam[valid_mask], y_cam[valid_mask], z_cam[valid_mask]

    if len(z_cam) == 0:
        return None

    # 픽셀 좌표로 투영
    u = fx * (x_cam / z_cam) + cx
    v = fy * (y_cam / z_cam) + cy

    # 정수 픽셀로 변환
    px = np.round(u).astype(int)
    py = np.round(v).astype(int)

    # 이미지 범위 내 좌표만 필터링
    in_bounds = (0 <= px) & (px < width) & (0 <= py) & (py < height)
    if not np.any(in_bounds):
        return None

    px, py = px[in_bounds], py[in_bounds]

    # 바운딩 박스 계산
    px_min, px_max = np.min(px), np.max(px)
    py_min, py_max = np.min(py), np.max(py)

    return (px_min, py_min, px_max, py_max)



def draw_red_bbox(image: np.ndarray, bbox: tuple, thickness: int = 2) -> np.ndarray:
    """
    이미지에 빨간색 BBox를 그리는 함수

    Parameters
    ----------
    image : np.ndarray
        입력 컬러 이미지 (H x W x 3)
    bbox : tuple or None
        (x_min, y_min, x_max, y_max)에 해당하는 픽셀 좌표,
        bbox가 None이면 직사각형을 그리지 않음

    Returns
    -------
    np.ndarray
        bbox가 그려진 새로운 이미지
    """
    # 이미지를 바로 수정하지 않고 복사본을 만들어 사용
    result_img = image.copy()

    if bbox is not None:
        height, width = result_img.shape[:2]
        (x_min, y_min, x_max, y_max) = bbox

        # 선이 바깥으로 나가지 않도록 마진 고려
        margin = thickness // 2

        # 클리핑: 두께만큼 안쪽으로
        x_min = max(margin, min(x_min, width - 1 - margin))
        y_min = max(margin, min(y_min, height - 1 - margin))
        x_max = max(margin, min(x_max, width - 1 - margin))
        y_max = max(margin, min(y_max, height - 1 - margin))

        # OpenCV에서 사각형 그리기
        cv2.rectangle(
            result_img,
            (x_min, y_min),
            (x_max, y_max),
            (255, 0, 0),  # BGR: 빨간색
            thickness          # 두께
        )
    return result_img



def stitch_bbox_images(images_list, savedir, idx_obj):
    subplot_size=512 
    n = min(9, len(images_list))
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1

    fig, axarr = plt.subplots(nrows, ncols, figsize=(5.12*ncols, 5.12*nrows), dpi=100, squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images_list[i])
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')

    plt.subplots_adjust(wspace=0.01, hspace=0.01)  # subplot 간격 제거
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)  # 여백 최소화
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # x축 눈금 제거
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # y축 눈금 제거

    plt.savefig(os.path.join(savedir, f"{idx_obj}.png"), dpi=100, pad_inches=0, facecolor="black")
    plt.close()



def save_debug_image(mask_list, dataset, save_root_dir, key, pcd):
    save_grid_images_dir = os.path.join(save_root_dir, 'grid')
    save_bbox_dir = os.path.join(save_root_dir, 'bbox')
    save_overlay_dir = os.path.join(save_root_dir, 'overlay')
    os.makedirs(save_grid_images_dir, exist_ok=True)
    os.makedirs(save_bbox_dir, exist_ok=True)
    os.makedirs(save_overlay_dir, exist_ok=True)

    image_list = []
    for mask_info in mask_list:
        frame_id, mask_id, conf = mask_info[0], mask_info[1], mask_info[2]
        rgb_path, seg_path = dataset.get_frame_path(frame_id)
        intrinsic_cam_parameters = dataset.get_intrinsics(frame_id)
        extrinsics = dataset.get_extrinsic(frame_id)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # temp_color_img, bbox = project(pcd, intrinsic_cam_parameters, extrinsics)
        bbox = get_bbox_by_projection(pcd, intrinsic_cam_parameters, extrinsics)
        bbox_image = draw_red_bbox(rgb, bbox)
        image_list.append(bbox_image)

        filename = f"{key}_{conf:.3f}_{frame_id}_.png"

        save_bbox_path = os.path.join(save_bbox_dir, filename)
        bbox_image_bgr = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_bbox_path, bbox_image_bgr)

        # mask = np.any(temp_color_img > 0, axis=2)
        # ovelay_image = draw_overlay_on_image(rgb, mask)
        # save_overlay_path = os.path.join(save_overlay_dir, filename)
        # ovelay_image_bgr = cv2.cvtColor(ovelay_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(save_overlay_path, ovelay_image_bgr)

    stitch_bbox_images(image_list, save_grid_images_dir, key)


def draw_overlay_on_image(rgb_image, mask):
    mask = cv2.resize(mask.astype(np.uint8), (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    left, top, right, bottom = mask2box(mask)

    output = rgb_image.copy()

    red_overlay = np.zeros_like(output, dtype=np.uint8)
    red_overlay[:, :] = (255, 0, 0)  # 빨간색 (RGB)

    mask_3ch = np.stack([mask]*3, axis=-1)

    alpha = 0.4
    output = np.where(mask_3ch, (1 - alpha) * output + alpha * red_overlay, output).astype(np.uint8)

    cv2.rectangle(output, (left, top), (right, bottom), (255, 0, 0), 2)
    return output


def main():
    args = get_args()
    args.seq_name_list = ['scene0000_00']  # 원하는 시퀀스 이름만 설정

    for seq_name in args.seq_name_list:
        args.seq_name = seq_name
        dataset = get_dataset(args)

        object_dict_path = f'{dataset.object_dict_dir}/{args.config}/object_dict.npy'
        object_dict = np.load(object_dict_path, allow_pickle=True).item()

        save_dir = os.path.join('bbox_outputs', seq_name)
        os.makedirs(save_dir, exist_ok=True)

        mesh = o3d.io.read_triangle_mesh(dataset.mesh_path)
        scene_points = np.asarray(mesh.vertices)

        pred = np.load('/workspace/MaskClustering/data/prediction/tasmap_class_agnostic/scene0000_00.npz')
        masks = pred['pred_masks']

        for key, value in tqdm(object_dict.items(), desc=f"[{seq_name}] Drawing BBoxes"):
            mask_list = value['mask_list'][:9]
            if len(mask_list) == 0:
                continue

            mask = masks[:, key]
            point_ids = np.where(mask)[0]
            points = scene_points[point_ids]

            color = (np.random.rand(3) * 0.7 + 0.3)
            colors = np.tile(color, (points.shape[0], 1))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            save_debug_image(mask_list, dataset, save_dir, key, pcd)

if __name__ == '__main__':
    main()
