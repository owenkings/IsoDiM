"""
IsoDiM - Motion Processing Utilities
====================================
Functions for motion data processing and visualization.
"""

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


# Kinematic chain definitions
kit_kinematic_chain = [
    [0, 11, 12, 13, 14, 15],
    [0, 16, 17, 18, 19, 20],
    [0, 1, 2, 3, 4],
    [3, 5, 6, 7],
    [3, 8, 9, 10]
]

t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20]
]

t2m_left_hand_chain = [
    [20, 22, 23, 24],
    [20, 34, 35, 36],
    [20, 25, 26, 27],
    [20, 31, 32, 33],
    [20, 28, 29, 30]
]

t2m_right_hand_chain = [
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 40, 41, 42],
    [21, 37, 38, 39],
    [21, 49, 50, 51]
]

# Raw joint offsets
kit_raw_offsets = np.array([
    [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
    [1, 0, 0], [0, -1, 0], [0, -1, 0], [-1, 0, 0], [0, -1, 0],
    [0, -1, 0], [1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
    [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, -1, 0], [0, 0, 1],
    [0, 0, 1]
])

t2m_raw_offsets = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [0, -1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0],
    [0, 0, 1], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
    [0, -1, 0], [0, -1, 0]
])


def qinv(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion inverse."""
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    
    Args:
        q: Quaternion tensor of shape (*, 4)
        v: Vector tensor of shape (*, 3)
        
    Returns:
        Rotated vectors of shape (*, 3)
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data: torch.Tensor):
    """
    Recover root rotation and position from motion data.
    
    Args:
        data: Motion data tensor
        
    Returns:
        Tuple of (root_rotation_quaternion, root_position)
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    
    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    
    return r_rot_quat, r_pos


def recover_from_ric(data: torch.Tensor, joints_num: int) -> torch.Tensor:
    """
    Recover 3D joint positions from RIC (Rotation-Invariant Coordinates).
    
    Args:
        data: Motion data in RIC format [B, T, D] or [T, D]
        joints_num: Number of joints (22 for HumanML3D, 21 for KIT)
        
    Returns:
        Joint positions [B, T, J, 3] or [T, J, 3]
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    
    # Add Y-axis rotation to local joints
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions
    )
    
    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    
    # Concatenate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    
    return positions


def plot_3d_motion(
    save_path: str,
    kinematic_tree: list,
    joints: np.ndarray,
    title: str,
    figsize: tuple = (10, 10),
    fps: int = 120,
    radius: float = 4
):
    """
    Plot and save 3D motion animation.
    
    Args:
        save_path: Path to save the animation
        kinematic_tree: Kinematic chain definition
        joints: Joint positions [T, J, 3]
        title: Plot title
        figsize: Figure size
        fps: Frames per second
        radius: Plot radius
    """
    matplotlib.use('Agg')
    
    # Format title for display
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([
            ' '.join(title_sp[:10]),
            ' '.join(title_sp[10:20]),
            ' '.join(title_sp[20:])
        ])
    elif len(title_sp) > 10:
        title = '\n'.join([
            ' '.join(title_sp[:10]),
            ' '.join(title_sp[10:])
        ])
    
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)
    
    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
    
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        'red', 'blue', 'black', 'red', 'blue',
        'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
        'darkred', 'darkred', 'darkred', 'darkred', 'darkred'
    ]
    frame_number = data.shape[0]
    
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1]
        )
        
        if index > 1:
            ax.plot3D(
                trajec[:index, 0] - trajec[index, 0],
                np.zeros_like(trajec[:index, 0]),
                trajec[:index, 1] - trajec[index, 1],
                linewidth=1.0, color='blue'
            )
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth, color=color
            )
        
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # Try different video encoding approaches
    save_success = False

    # First try: save as GIF (most reliable)
    try:
        gif_path = save_path.replace('.mp4', '.gif')
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps)
        ani.save(gif_path, writer=writer)
        print(f"Video saved as GIF: {gif_path}")
        save_success = True
    except Exception as e_gif:
        print(f"GIF attempt failed: {e_gif}")

    # Second try: save as MP4 if GIF failed
    if not save_success:
        try:
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, codec='libopenh264', bitrate=1800)
            ani.save(save_path, writer=writer)
            print(f"Video saved as MP4: {save_path}")
            save_success = True
        except Exception as e_mp4:
            print(f"MP4 attempt failed: {e_mp4}")

    # Final fallback: try default ffmpeg save
    if not save_success:
        try:
            ani.save(save_path, fps=fps, writer='ffmpeg')
            print(f"Video saved with default method: {save_path}")
            save_success = True
        except Exception as e_default:
            print(f"All video saving attempts failed: {e_default}")

    if not save_success:
        raise RuntimeError("Could not save video with any available method")

    plt.close()

