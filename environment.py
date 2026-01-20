import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RiskSensitiveGridWorld(gym.Env):
    """
    Risky Bridge Gridworld (POMDP) — RANDOM START VERSION (Option A)

    Two paths:
      - Safe path -> lower reward, no risk
      - Risky path -> higher reward, one-shot catastrophic risk

    Randomizing the start position breaks the strict coupling between
    timestep and commitment, enabling time-matched probing.

    IMPORTANT NOTES:
      - Hazard channel is NOT shown in obs.
      - The environment is still partially observable.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        slip_prob: float = 0.1,
        max_steps: int = 100,
        random_start: bool = True,
        shaping_coef: float = 0.05,
        stuck_penalty: float = 0.05,
    ):
        super().__init__()

        self.width = 15
        self.height = 7

        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.random_start = random_start

        # shaping
        self.shaping_coef = shaping_coef
        self.stuck_penalty = stuck_penalty

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Obs: 3×5×5
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, 5, 5), dtype=np.float32
        )

        # Define valid start states (kept small + safe for training stability)
        self.start_positions = [
            (3, 1), (3, 2), (3, 3), (3, 4),
            (4, 1), (4, 2), (4, 3),
            # (5, 1), (5, 2),  # can start near hazard row
            (1, 1), (1, 2), (1, 3), (1, 4)
        ]

        # fixed goal positions
        self.risky_goal = np.array([5, 9], dtype=np.int32)
        self.safe_goal  = np.array([4, 13], dtype=np.int32)

        self.reset()

    # helpers

    def _manhattan(self, a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    # Core API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start:
            while True:
                r, c = self.start_positions[np.random.randint(len(self.start_positions))]
                if not self._is_wall((r, c)):
                    self.agent_pos = np.array([r, c], dtype=np.int32)
                    break
        else:
            self.agent_pos = np.array([3, 1], dtype=np.int32)

        self.current_step = 0
        self.used_risky = False
        self.risk_resolved = False

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # store old position for shaping
        prev_pos = self.agent_pos.copy()

        # Which target should we pursue?
        # (based on internal commitment flag, NOT hazard visibility)
        prev_target = self.risky_goal if self.used_risky else self.safe_goal
        prev_dist = self._manhattan(prev_pos, prev_target)

        # movement
        moves = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }

        new_pos = self.agent_pos + moves[action]
        if not self._is_wall(new_pos):
            self.agent_pos = new_pos

        reward = -0.1
        terminated = False
        truncated = self.current_step >= self.max_steps

        # discourage "bumping wall / no-op loops"
        if np.array_equal(self.agent_pos, prev_pos):
            reward -= self.stuck_penalty

        # One-shot risk 
        if self._on_hazard(self.agent_pos) and not self.risk_resolved:
            self.used_risky = True
            self.risk_resolved = True
            if np.random.rand() < self.slip_prob:
                reward -= 50.0
                terminated = True

        # Risky goal 
        if tuple(self.agent_pos) == tuple(self.risky_goal):
            reward += 65.0
            terminated = True

        # Safe goal 
        if tuple(self.agent_pos) == tuple(self.safe_goal):
            reward += 20.0
            terminated = True

       
        # After commitment, shaping pulls toward risky goal.
        # Otherwise, it pulls toward safe goal.
        target = self.risky_goal if self.used_risky else self.safe_goal
        new_dist = self._manhattan(self.agent_pos, target)

        progress = prev_dist - new_dist  # +1 if closer, -1 if farther
        reward += self.shaping_coef * float(progress)

        return self._get_obs(), reward, terminated, truncated, {}

    # Environment structure

    def _is_wall(self, pos):
        r, c = pos

        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return True

        # boundary walls
        if r in [0, 6] or c in [0, 14]:
            return True

        # internal walls
        if r in [2, 4] and 2 <= c <= 12:
            return True

        return False

    def _on_hazard(self, pos):
        r, c = pos
        return r == 5 and 2 <= c <= 4

    # Observation model
    

    def _get_obs(self):
        full_map = np.zeros((3, self.height, self.width), dtype=np.float32)

        # Walls
        for r in range(self.height):
            for c in range(self.width):
                if self._is_wall((r, c)):
                    full_map[0, r, c] = 1.0

        # Goals
        full_map[1, 5, 9] = 1.0
        full_map[1, 4, 13] = 1.0

        # Hazards channel will be turned of during training (partially observable), so that information gets leaked
        for c in range(2, 5):
            full_map[2, 5, c] = 1.0

        padded = np.pad(
            full_map,
            pad_width=((0, 0), (2, 2), (2, 2)),
            mode="constant",
            constant_values=0.0,
        )

        r, c = self.agent_pos
        r += 2
        c += 2

        return padded[:, r - 2: r + 3, c - 2: c + 3].astype(np.float32)

    # Utilities

    def valid_actions(self):
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

        valid = []
        for a, (dr, dc) in moves.items():
            new_pos = self.agent_pos + np.array([dr, dc])
            if not self._is_wall(new_pos):
                valid.append(a)
        return valid

    def render_ascii(self):
        symbols = {"wall": "#", "empty": ".", "agent": "A", "goal": "G", "hazard": "~"}
        grid = [[symbols["empty"] for _ in range(self.width)]
                for _ in range(self.height)]

        for r in range(self.height):
            for c in range(self.width):
                if self._is_wall((r, c)):
                    grid[r][c] = symbols["wall"]

        grid[5][9] = symbols["goal"]
        grid[4][13] = symbols["goal"]

        for c in range(2, 5):
            grid[5][c] = symbols["hazard"]

        ar, ac = self.agent_pos
        grid[ar][ac] = symbols["agent"]

        print("\n".join(" ".join(row) for row in grid))
