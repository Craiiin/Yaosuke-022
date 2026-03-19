import taichi as ti
import math

# 初始化 Taichi，CPU/GPU 均可（GPU 渲染更快）
ti.init(arch=ti.cpu)  # 如需GPU加速，改为 ti.init(arch=ti.gpu)

# ===================== 1. 定义数据结构 =====================
# 立方体顶点：8个3D顶点（x,y,z）
cube_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
# 变换后的屏幕坐标：8个2D坐标（x,y）
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)
# 立方体棱边：12条边，每条边对应两个顶点的索引
edges = ti.Vector.field(2, dtype=ti.i32, shape=12)

# ===================== 2. MVP 变换矩阵（复用你之前的逻辑） =====================
@ti.func
def get_model_matrix(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32):
    """模型变换矩阵：绕 X/Y/Z 轴旋转（支持立方体多轴旋转）"""
    # 绕X轴旋转矩阵
    rad_x = angle_x * math.pi / 180.0
    cx, sx = ti.cos(rad_x), ti.sin(rad_x)
    rot_x = ti.Matrix([
        [1.0, 0.0,  0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx,  cx, 0.0],
        [0.0, 0.0,  0.0, 1.0]
    ])
    # 绕Y轴旋转矩阵
    rad_y = angle_y * math.pi / 180.0
    cy, sy = ti.cos(rad_y), ti.sin(rad_y)
    rot_y = ti.Matrix([
        [cy,  0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # 绕Z轴旋转矩阵
    rad_z = angle_z * math.pi / 180.0
    cz, sz = ti.cos(rad_z), ti.sin(rad_z)
    rot_z = ti.Matrix([
        [cz, -sz, 0.0, 0.0],
        [sz,  cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # 组合旋转矩阵（先绕Z→再绕Y→最后绕X）
    return rot_x @ rot_y @ rot_z

@ti.func
def get_view_matrix(eye_pos):
    """视图变换矩阵：相机位置（固定在Z轴正方向）"""
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """透视投影矩阵（复用你之前的实现）"""
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    # 透视挤压矩阵
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    # 正交投影矩阵
    M_ortho_scale = ti.Matrix([
        [2.0/(r-l), 0.0, 0.0, 0.0],
        [0.0, 2.0/(t-b), 0.0, 0.0],
        [0.0, 0.0, 2.0/(n-f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r+l)/2.0],
        [0.0, 1.0, 0.0, -(t+b)/2.0],
        [0.0, 0.0, 1.0, -(n+f)/2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

# ===================== 3. 并行计算坐标变换 =====================
@ti.kernel
def compute_transform(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32):
    """计算所有顶点的3D→2D变换"""
    # 相机位置：Z轴正方向8个单位（远离立方体）
    eye_pos = ti.Vector([0.0, 0.0, 8.0])
    # 组合MVP矩阵
    model = get_model_matrix(angle_x, angle_y, angle_z)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(60.0, 1.0, 0.1, 100.0)
    mvp = proj @ view @ model

    # 对每个顶点执行变换
    for i in range(8):
        v = cube_vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])  # 补全齐次坐标
        v_clip = mvp @ v4                        # MVP变换
        v_ndc = v_clip / v_clip[3]               # 透视除法（NDC坐标）
        # 视口变换：映射到GUI窗口[0,1]范围
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

# ===================== 4. 初始化立方体 =====================
def init_cube():
    """初始化立方体的顶点和棱边"""
    # 1. 立方体顶点（中心在原点，边长为2）
    cube_vertices_np = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # 后平面（Z=-1）
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]    # 前平面（Z=1）
    ]
    # 赋值给Taichi field
    for i in range(8):
        cube_vertices[i] = cube_vertices_np[i]
    
    # 2. 立方体棱边（12条边，对应顶点索引）
    edges_np = [
        [0,1], [1,2], [2,3], [3,0],  # 后平面4条边
        [4,5], [5,6], [6,7], [7,4],  # 前平面4条边
        [0,4], [1,5], [2,6], [3,7]   # 前后平面连接的4条边
    ]
    for i in range(12):
        edges[i] = edges_np[i]

# ===================== 5. 主函数（可视化+交互） =====================
def main():
    # 初始化立方体
    init_cube()
    
    # 创建GUI窗口（800x800分辨率）
    gui = ti.GUI("3D Cube Visualization (Taichi)", res=(800, 800), background_color=0x111111)
    
    # 旋转角度（初始为0）
    angle_x, angle_y, angle_z = 0.0, 0.0, 0.0
    
    # 主循环
    while gui.running:
        # ========== 交互控制 ==========
        if gui.get_event(ti.GUI.PRESS):
            # 按键控制旋转（X/Y/Z轴）
            if gui.event.key == 'w': angle_x += 5.0   # W：绕X轴顺时针
            elif gui.event.key == 's': angle_x -= 5.0 # S：绕X轴逆时针
            elif gui.event.key == 'a': angle_y += 5.0 # A：绕Y轴顺时针
            elif gui.event.key == 'd': angle_y -= 5.0 # D：绕Y轴逆时针
            elif gui.event.key == 'q': angle_z += 5.0 # Q：绕Z轴顺时针
            elif gui.event.key == 'e': angle_z -= 5.0 # E：绕Z轴逆时针
            elif gui.event.key == ti.GUI.ESCAPE: gui.running = False # ESC退出
        
        # 鼠标拖动控制旋转（更流畅的交互）
        if gui.is_pressed(ti.GUI.LMB):  # 按住左键拖动
            dx, dy = gui.get_cursor_delta()
            angle_y += dx * 100  # 鼠标X方向 → 绕Y轴旋转
            angle_x -= dy * 100  # 鼠标Y方向 → 绕X轴旋转
        
        # ========== 计算变换 ==========
        compute_transform(angle_x, angle_y, angle_z)
        
        # ========== 绘制立方体 ==========
        # 遍历所有棱边，绘制每条边
        for i in range(12):
            v1_idx = edges[i][0]
            v2_idx = edges[i][1]
            # 获取两个顶点的2D坐标
            p1 = screen_coords[v1_idx]
            p2 = screen_coords[v2_idx]
            # 绘制棱边（白色，线宽2）
            gui.line(p1, p2, radius=2, color=0xFFFFFF)
        
        # 显示提示文字
        gui.text("W/S: X轴 | A/D: Y轴 | Q/E: Z轴 | 鼠标左键拖动", (0.05, 0.05), color=0x888888)
        
        # 刷新窗口
        gui.show()

if __name__ == '__main__':
    main()