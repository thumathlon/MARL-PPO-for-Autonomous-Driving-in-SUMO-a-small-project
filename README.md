# MARL-PPO-for-Traffic-Optimization-in-SUMO

🚗 An advanced implementation of **Multi-Agent Reinforcement Learning (MARL)** using **Proximal Policy Optimization (PPO)** to train autonomous vehicles in the **SUMO** traffic simulator. This project focuses on optimizing a complex highway ramp-merging scenario to enhance traffic efficiency, safety, and fairness.


---

## 🌟 Key Features

* **Advanced RL Algorithm**: Implements a robust **Multi-Agent PPO** agent capable of handling continuous action spaces for realistic vehicle control.
* **Complex Driving Scenario**: Utilizes the SUMO simulator to model a challenging highway merge, a critical bottleneck in real-world traffic.
* **Sophisticated Hybrid Reward Function**: The environment (`sumo_env.py`) uses a carefully designed reward structure that balances:
    * **🚗 Individual Rewards**: Encourages agents to drive at the speed limit while maintaining comfort (low acceleration/jerk) and avoiding dangerously low speeds.
    * **🌍 Social Rewards**: Promotes system-level efficiency by rewarding average network speed and heavily penalizing collisions.
* **Effective Training Strategy**: The training script (`train.py`) incorporates modern deep RL techniques:
    * Learning Rate and Exploration Noise Decay
    * Early Stopping with Patience
    * Detailed logging of rewards for analysis.
* **Comprehensive Evaluation Dashboard**: The `evaluate.py` script provides a clear and holistic performance assessment based on:
    * **Efficiency**: Success Rate, Average Travel Time, and System Throughput.
    * **Safety**: Total Collisions and Emergency Brake Events.
    * **Fairness**: **Jain's Fairness Index** applied to vehicle travel times.

---

## 🛠️ Tech Stack

* **Frameworks**: PyTorch, OpenAI Gym
* **Simulator**: SUMO (Simulation of Urban MObility)
* **Algorithm**: Proximal Policy Optimization (PPO)
* **Libraries**: NumPy, Matplotlib

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* SUMO installed and `SUMO_HOME` environment variable configured.
* PyTorch

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/thumathlon/MARL-PPO-for-Traffic-Optimization-in-SUMO.git](https://github.com/thumathlon/MARL-PPO-for-Traffic-Optimization-in-SUMO.git)
    cd MARL-PPO-for-Traffic-Optimization-in-SUMO
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**1. Train the Model:**
To start training the PPO agent, run:
```bash
python train.py
```
Training progress, logs, and learning curve plots will be saved locally. The best-performing model will be saved as `MARL_PPO_SUMO_Optimized_best.pth`.

**2. Evaluate the Trained Model:**
To evaluate the performance of your saved model, run the evaluation script:
```bash
python evaluate.py
```
This will run the simulation for multiple episodes without training and print a final "Evaluation Dashboard" with all the key metrics.

---

## 🙏 How to Contribute

Contributions are welcome! If you have ideas for improvements or find any issues, please open an issue or submit a pull request.

## ⭐ Star This Project

If you find this project useful or interesting, please give it a star! It helps to grow the community and shows your appreciation for the work.
