import tensorflow as tf
import math
import json
import os

class CustomCosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, first_decay_steps, warmup_steps=100, t_mul=2.0, m_mul=1.0, alpha=0.0, offset=0, name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.warmup_steps = warmup_steps
        self.t_mul = tf.cast(t_mul, tf.float32)
        self.m_mul = m_mul
        self.alpha = alpha
        self.name = name
        self.offset = offset
        self.current_learning = initial_learning_rate

    def __call__(self, step):
        step = step + self.offset
        step = tf.cast(step, tf.float32)

        if step < self.warmup_steps:
            self.current_learning = (self.initial_learning_rate / tf.cast(self.warmup_steps, tf.float32)) * step
            return self.current_learning

        step = step - self.warmup_steps
        completed_fraction = step / tf.cast(self.first_decay_steps, tf.float32)
        i_restart = tf.floor(tf.math.log(1.0 - completed_fraction * (1.0 - self.t_mul)) / tf.math.log(self.t_mul))
        sum_r = (1.0 - self.t_mul**i_restart) / (1.0 - self.t_mul)
        completed_fraction = (step / tf.cast(self.first_decay_steps, tf.float32) - sum_r) / self.t_mul**i_restart
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        self.current_learning = tf.convert_to_tensor(self.initial_learning_rate * decayed * (self.m_mul**i_restart), dtype=tf.float32)
        return self.current_learning

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "warmup_steps": self.warmup_steps,
            "t_mul": self.t_mul,
            "m_mul": self.m_mul,
            "alpha": self.alpha,
            "name": self.name
        }

    def save(self, filepath, epoch, num_steps):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "initial_learning_rate": float(self.initial_learning_rate),
            "first_decay_steps": int(self.first_decay_steps),
            "warmup_steps": int(self.warmup_steps),
            "t_mul": float(self.t_mul),
            "m_mul": float(self.m_mul),
            "alpha": float(self.alpha),
            "offset": int(epoch*num_steps)
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        return cls(
            initial_learning_rate=state["initial_learning_rate"],
            first_decay_steps=state["first_decay_steps"],
            warmup_steps=state["warmup_steps"],
            t_mul=state["t_mul"],
            m_mul=state["m_mul"],
            alpha=state["alpha"],
            offset=state["offset"]
        )

"""

import matplotlib.pyplot as plt

# Create an instance of the custom learning rate schedule
initial_learning_rate = 0.1
first_decay_steps = 1000
warmup_steps = 100
t_mul = 2.0
m_mul = 1.0
alpha = 0.0

lr_schedule = CustomCosineDecayRestarts(
    initial_learning_rate=initial_learning_rate,
    first_decay_steps=first_decay_steps,
    warmup_steps=warmup_steps,
    t_mul=t_mul,
    m_mul=m_mul,
    alpha=alpha
)

# Simulate training steps
steps = 5000
learning_rates = []

for step in range(steps):
    lr = lr_schedule(step)
    learning_rates.append(lr)

# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(range(steps), learning_rates)
plt.title('Custom Cosine Decay with Restarts Learning Rate Schedule')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.savefig('/home/amarinai/DeepLearningThesis/BCFind-v2/bcfind/scheduler/scheduler_plot.png')
"""
