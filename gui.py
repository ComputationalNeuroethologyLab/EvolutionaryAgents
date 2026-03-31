import tkinter as tk
from tkinter import filedialog, messagebox
import math
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import *
from environment import VisualAgent, Circle, Line

class SimulationGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Visual Agent Simulation")
        
        self.agent = VisualAgent()
        self.obj = Circle()
        
        self.is_running = False
        self.loop_id = None
        self.drag_data = {"item": None, "was_running": False}
        
        # Simulation playback speed multiplier
        self.sim_speed = 1.0
        
        # 5 seconds of history at 0.1 step size = 50 steps
        self.history_len = int(5.0 / STEP_SIZE)
        self.output_history = {}
        self.neuron_vars = []
        self.neuron_colors = []
        
        self.setup_ui()
        self.reset_sim()

    def setup_ui(self):
        # Top Control Bar
        btn_frame = tk.Frame(self.master)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(btn_frame, text="Load Network (.ns)", command=self.load_network).pack(side=tk.LEFT, padx=5)
        self.play_btn = tk.Button(btn_frame, text="Play", command=self.play_pause)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Stop / Reset", command=self.reset_sim).pack(side=tk.LEFT, padx=5)
        
        # Stimulus Toggle
        self.stim_btn = tk.Button(btn_frame, text="Switch to Line", command=self.toggle_stimulus)
        self.stim_btn.pack(side=tk.LEFT, padx=5)

        # Simulation Speed Slider
        self.speed_slider = tk.Scale(btn_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                     label="Sim Speed (x)", command=self.update_speed, length=150)
        self.speed_slider.set(self.sim_speed)
        self.speed_slider.pack(side=tk.LEFT, padx=10)

        # Widened Velocity Slider
        self.vel_slider = tk.Scale(btn_frame, from_=-10.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                   label="Obj Velocity Y", command=self.update_vel, length=200)
        self.vel_slider.set(self.obj.vy)
        self.vel_slider.pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar(value="Status: Network not loaded.")
        tk.Label(btn_frame, textvariable=self.status_var).pack(side=tk.RIGHT, padx=10)

        # Main Layout (Left: Canvas, Right: Plot & Checks)
        main_pane = tk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas Setup
        self.canvas_width = 450
        self.canvas_height = 350
        self.canvas = tk.Canvas(main_pane, width=self.canvas_width, height=self.canvas_height, bg="white")
        main_pane.add(self.canvas)

        # Matplotlib & Checkbox setup
        right_frame = tk.Frame(main_pane)
        main_pane.add(right_frame)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_title("Neural Dynamics (Last 5s)")
        self.ax.set_ylim(0, 1.05)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.check_frame = tk.Frame(right_frame)
        self.check_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def update_speed(self, val):
        self.sim_speed = float(val)

    def update_vel(self, val):
        self.obj.vy = float(val)

    def toggle_stimulus(self):
        old_cx, old_cy, old_vy = self.obj.cx, self.obj.cy, self.obj.vy
        if isinstance(self.obj, Circle):
            self.obj = Line(cx=old_cx, cy=old_cy, vy=old_vy)
            self.stim_btn.config(text="Switch to Circle")
        else:
            self.obj = Circle(cx=old_cx, cy=old_cy, vy=old_vy)
            self.stim_btn.config(text="Switch to Line")
        self.draw_environment()

    def load_network(self):
        filepath = filedialog.askopenfilename(filetypes=[("NS files", "*.ns"), ("All files", "*.*")])
        if filepath:
            success = self.agent.ctrnn.load_from_file(filepath)
            if success:
                self.status_var.set(f"Status: Loaded CTRNN (Size {self.agent.ctrnn.size})")
                self.setup_checkboxes()
                self.reset_sim()
            else:
                messagebox.showerror("Error", "Failed to parse .ns file. Check formatting.")

    def setup_checkboxes(self):
        for widget in self.check_frame.winfo_children():
            widget.destroy()
        
        self.neuron_vars = []
        self.neuron_colors = []
        self.output_history = {i: deque(maxlen=self.history_len) for i in range(self.agent.ctrnn.size)}
        
        row, col = 0, 0
        for i in range(self.agent.ctrnn.size):
            color_hex = mcolors.to_hex(plt.cm.tab20(i % 20))
            self.neuron_colors.append(color_hex)
            
            var = tk.BooleanVar(value=True)
            self.neuron_vars.append(var)
            
            cb = tk.Checkbutton(self.check_frame, text=f"N{i}", variable=var, fg=color_hex)
            cb.grid(row=row, column=col, sticky="w")
            
            col += 1
            if col > 6:
                col = 0
                row += 1

    def play_pause(self):
        if self.agent.ctrnn.size == 0:
            messagebox.showwarning("Warning", "Please load a valid .ns file first!")
            return
            
        self.is_running = not self.is_running
        self.play_btn.config(text="Pause" if self.is_running else "Play")
        
        if self.is_running:
            self.run_loop()
        else:
            if self.loop_id is not None:
                self.master.after_cancel(self.loop_id)
                self.loop_id = None

    def reset_sim(self):
        self.is_running = False
        self.play_btn.config(text="Play")
        if self.loop_id is not None:
            self.master.after_cancel(self.loop_id)
            self.loop_id = None
            
        self.agent.reset()
        self.obj.reset()
        for i in range(self.agent.ctrnn.size):
            if i in self.output_history:
                self.output_history[i].clear()
            
        self.draw_environment()
        self.update_plot()

    def run_loop(self):
        if self.is_running:
            if self.obj.cy > BODY_SIZE / 2:
                self.agent.step(STEP_SIZE, self.obj)
                self.obj.step(STEP_SIZE)
                
                for i in range(self.agent.ctrnn.size):
                    self.output_history[i].append(self.agent.ctrnn.outputs[i])
                
                self.draw_environment()
                self.update_plot()
                
                # Calculate new delay based on standard 20ms baseline modified by sim_speed
                # Use max(1, ...) so Tkinter never receives a delay of 0 milliseconds
                delay = max(1, int(20 / self.sim_speed))
                self.loop_id = self.master.after(delay, self.run_loop)
            else:
                self.is_running = False
                self.play_btn.config(text="Play")
                self.status_var.set("Status: Simulation Complete")
                self.loop_id = None

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Neural Dynamics (Last 5s)")
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_xlim(0, max(1, self.history_len)) 
        
        for i, var in enumerate(self.neuron_vars):
            if var.get() and len(self.output_history[i]) > 0:
                self.ax.plot(self.output_history[i], color=self.neuron_colors[i], alpha=0.2, label=f'N{i}')
                
        self.plot_canvas.draw()

    # --- Coordinates & Drawing ---
    def coord_to_px(self, x, y):
        px = x + (self.canvas_width / 2)
        py = self.canvas_height - y - 25 
        return px, py

    def px_to_coord(self, px, py):
        x = px - (self.canvas_width / 2)
        y = self.canvas_height - py - 25
        return x, y

    def on_press(self, event):
        cx_px, cy_px = self.coord_to_px(self.obj.cx, self.obj.cy)
        radius_px = self.obj.size / 2 
        
        if (event.x - cx_px)**2 + (event.y - cy_px)**2 <= radius_px**2:
            self.drag_data["item"] = "object"
            self.drag_data["was_running"] = self.is_running
            if self.is_running:
                self.play_pause()

    def on_drag(self, event):
        if self.drag_data["item"] == "object":
            new_x, new_y = self.px_to_coord(event.x, event.y)
            self.obj.cx = new_x
            self.obj.cy = new_y
            self.agent.calculate_rays()
            self.draw_environment()

    def on_release(self, event):
        if self.drag_data["item"] == "object":
            self.drag_data["item"] = None
            if self.drag_data["was_running"]:
                self.play_pause()

    def draw_environment(self):
        self.canvas.delete("all")
        
        _, ground_y = self.coord_to_px(0, 0)
        self.canvas.create_line(0, ground_y, self.canvas_width, ground_y, fill="black", width=2)
        
        if self.agent.rays:
            for ray in self.agent.rays:
                if ray['m'] == float('inf'):
                    end_x = ray['startX']
                    end_y = ray['startY'] + ray['length']
                else:
                    dx = ray['length'] / math.sqrt(1 + ray['m']**2)
                    end_x = ray['startX'] + (dx if math.sin(math.atan2(1, ray['m'])) > 0 else -dx)
                    if ray['startX'] < self.agent.cx:
                        end_x = ray['startX'] - dx
                    else:
                        end_x = ray['startX'] + dx
                    if abs(ray['startX'] - self.agent.cx) < 1e-5:
                        end_x = ray['startX']
                    end_y = ray['m'] * end_x + ray['b']

                px_start, py_start = self.coord_to_px(ray['startX'], ray['startY'])
                px_end, py_end = self.coord_to_px(end_x, end_y)
                self.canvas.create_line(px_start, py_start, px_end, py_end, fill="green", dash=(2,2))

        ax1, ay1 = self.coord_to_px(self.agent.cx - BODY_SIZE/2, self.agent.cy - BODY_SIZE/2)
        ax2, ay2 = self.coord_to_px(self.agent.cx + BODY_SIZE/2, self.agent.cy + BODY_SIZE/2)
        self.canvas.create_arc(ax1, ay1, ax2, ay2, start=0, extent=180, fill="blue")

        if self.obj.type == "circle":
            ox1, oy1 = self.coord_to_px(self.obj.cx - self.obj.size/2, self.obj.cy + self.obj.size/2)
            ox2, oy2 = self.coord_to_px(self.obj.cx + self.obj.size/2, self.obj.cy - self.obj.size/2)
            self.canvas.create_oval(ox1, oy1, ox2, oy2, fill="red")
        elif self.obj.type == "line":
            lx1, ly1 = self.coord_to_px(self.obj.cx - self.obj.size/2, self.obj.cy)
            lx2, ly2 = self.coord_to_px(self.obj.cx + self.obj.size/2, self.obj.cy)
            self.canvas.create_line(lx1, ly1, lx2, ly2, fill="red", width=5)
