
# manim -pql Manim/conv.py ConvolutionOperation

from manim import *
import numpy as np

class ConvolutionOperation(Scene):
    def construct(self):
        # Parameters
        image_size, filter_size = 5, 3
        stride, padding = 1, 0
        output_size = (image_size - filter_size + 2 * padding) // stride + 1

        # Data
        np.random.seed(42)
        image_values = np.round(np.random.rand(image_size, image_size) * 9)
        filter_values = np.round(np.random.rand(filter_size, filter_size) * 4) - 2

        # Create grids
        input_grid = self.create_grid(image_size, image_values, 0.7)
        filter_grid = self.create_grid(filter_size, filter_values, 0.5, color=BLUE)
        output_grid = self.create_empty_grid(output_size, 0.7, color=GREEN)

        # Titles
        input_title = Text("Input Image", font_size=32)
        filter_title = Text("Filter", font_size=28, color=BLUE)
        output_title = Text("Feature Map", font_size=32, color=GREEN)

        # Position elements
        input_group = VGroup(input_title, input_grid).arrange(DOWN, buff=0.3)
        filter_group = VGroup(filter_title, filter_grid).arrange(DOWN, buff=0.3)
        output_group = VGroup(output_title, output_grid).arrange(DOWN, buff=0.3)

        VGroup(input_group, filter_group, output_group).arrange(RIGHT, buff=2.5)
        input_group.to_edge(LEFT)

        # Formula
        formula = MathTex(
            r"\text{Feature}[i,j] = \sum_{m,n} \text{Input}[i+m,j+n] \times \text{Filter}[m,n]",
            font_size=40
        ).to_edge(UP)

        # Show formula and elements
        self.play(Write(formula))
        self.wait(0.2)
        self.play(FadeIn(input_group, filter_group, output_group), run_time=2)
        self.wait(1)

        # Create filter window rectangle
        window = SurroundingRectangle(
            VGroup(*[input_grid.get_cell(0+m, 0+n) for m in range(filter_size) for n in range(filter_size)]),
            color=YELLOW,
            buff=0.1
        )
        self.play(Create(window))
        self.wait(0.5)

        # Convolution Operation
        result_texts = {}
        for i in range(output_size):
            for j in range(output_size):
                # Move window
                new_cells = VGroup(*[input_grid.get_cell(i+m, j+n) for m in range(filter_size) for n in range(filter_size)])
                new_window = SurroundingRectangle(new_cells, color=YELLOW, buff=0.1)
                self.play(Transform(window, new_window), run_time=0.4)

                # Highlight cells
                highlights = []
                for m in range(filter_size):
                    for n in range(filter_size):
                        highlights.append(input_grid.get_cell(i+m, j+n))
                        highlights.append(filter_grid.get_cell(m, n))
                self.play(*[cell.animate.set_fill(YELLOW, opacity=0.5) for cell in highlights], run_time=0.3)

                # Build operation string
                operation_parts = []
                total_sum = 0
                for m in range(filter_size):
                    for n in range(filter_size):
                        val_in = int(image_values[i+m, j+n])
                        val_filt = int(filter_values[m, n])
                        mult_result = val_in * val_filt
                        total_sum += mult_result
                        operation_parts.append(f"({val_in} \\times {val_filt})")

                full_operation = " + ".join(operation_parts) + f" = {total_sum}"

                if len(full_operation) > 80:
                    mid = len(operation_parts) // 2
                    upper = " + ".join(operation_parts[:mid])
                    lower = " + ".join(operation_parts[mid:])
                    operation_text = MathTex(upper + r"\\" + lower + f"= {total_sum}", font_size=30)
                else:
                    operation_text = MathTex(full_operation, font_size=32)

                operation_text.move_to(DOWN * 2.5)

                self.play(FadeIn(operation_text), run_time=0.4)

                # Show the result
                result = Text(f"{total_sum:.1f}", font_size=20)
                result.move_to(output_grid.get_cell_center(i, j))
                self.play(FadeIn(result), run_time=0.4)
                result_texts[(i, j)] = result

                # Clear operation
                self.play(FadeOut(operation_text), run_time=0.3)

                # Unhighlight
                self.play(*[cell.animate.set_fill(opacity=0) for cell in highlights], run_time=0.2)

        # Clean up
        self.play(FadeOut(window))
        self.wait(0.5)

        # Final Celebration
        all_results = VGroup(*result_texts.values())
        self.play(all_results.animate.set_color(GREEN), run_time=0.2)
        self.wait(0.5)

        # =====================
        # New Frame: Advantages
        # =====================
        self.play(
        FadeOut(input_group),
        FadeOut(filter_group),
        FadeOut(output_group),
        FadeOut(formula),
        FadeOut(all_results),    )
        self.wait(0.5)

        # self.play(FadeOut(input_group, filter_group, output_group, formula))
        # self.wait(0.5)

        title = Text("Why Convolution?", font_size=48, color=YELLOW).to_edge(UP)

        points = BulletedList(
            "Reduces parameters via weight sharing",
            "Detects local patterns (edges, textures)",
            "Translation invariance (moves with object)",
            "Learns filters automatically from data",
            "Stacking filters → higher-level features",
            dot_scale_factor=1.2,
            font_size=32
        ).scale(0.9).next_to(title, DOWN, buff=0.8)

        extra_info = Text(
            "Convolutional Neural Networks (CNNs)\nuse many layers of convolutions!",
            font_size=30, color=BLUE
        ).next_to(points, DOWN, buff=1)

        self.play(Write(title))
        self.wait(0.3)
        self.play(FadeIn(points, extra_info), run_time=2)
        self.wait(4)

    def create_grid(self, size, values, scale=1, color=WHITE):
        grid = VGroup()
        for i in range(size):
            for j in range(size):
                square = Square(side_length=scale)
                square.set_stroke(color, 2)
                square.move_to(np.array([j * scale, -i * scale, 0]))
                number = Text(f"{values[i, j]:.0f}", font_size=24)
                number.move_to(square.get_center())
                grid.add(VGroup(square, number))
        grid.get_cell = lambda i, j: grid[i * size + j][0]
        grid.get_cell_center = lambda i, j: grid[i * size + j][0].get_center()
        return grid

    def create_empty_grid(self, size, scale=1, color=WHITE):
        grid = VGroup()
        for i in range(size):
            for j in range(size):
                square = Square(side_length=scale)
                square.set_stroke(color, 2)
                square.move_to(np.array([j * scale, -i * scale, 0]))
                grid.add(VGroup(square))
        grid.get_cell = lambda i, j: grid[i * size + j][0]
        grid.get_cell_center = lambda i, j: grid[i * size + j][0].get_center()
        return grid



# manim -pql Manim/conv.py ConvolutionOperationOnlyConv


from manim import *
import numpy as np

class ConvolutionOperationOnlyConv(Scene):
    def construct(self):
        # self.camera.background_color = GRAY
        # Parameters
        image_size, filter_size = 5, 3
        stride, padding = 1, 0
        output_size = (image_size - filter_size + 2 * padding) // stride + 1

        # Data
        np.random.seed(42)
        image_values = np.round(np.random.rand(image_size, image_size) * 9)
        filter_values = np.round(np.random.rand(filter_size, filter_size) * 4) - 2

        # Create grids
        input_grid = self.create_grid(image_size, image_values, 0.7)
        filter_grid = self.create_grid(filter_size, filter_values, 0.5, color=BLUE)
        output_grid = self.create_empty_grid(output_size, 0.7, color=GREEN)

        # Titles
        input_title = Text("Input Image", font_size=32)
        filter_title = Text("Filter", font_size=28, color=BLUE)
        output_title = Text("Feature Map", font_size=32, color=GREEN)

        # Position elements
        input_group = VGroup(input_title, input_grid).arrange(DOWN, buff=0.3)
        filter_group = VGroup(filter_title, filter_grid).arrange(DOWN, buff=0.3)
        output_group = VGroup(output_title, output_grid).arrange(DOWN, buff=0.3)

        VGroup(input_group, filter_group, output_group).arrange(RIGHT, buff=2.5)
        input_group.to_edge(LEFT)

        # Formula
        formula = MathTex(
            r"\text{Feature}[i,j] = \sum_{m,n} \text{Input}[i+m,j+n] \times \text{Filter}[m,n]",
            font_size=40
        ).to_edge(UP)

        # Show formula and elements
        self.play(Write(formula))
        self.wait(0.2)
        self.play(FadeIn(input_group, filter_group, output_group), run_time=1)
        self.wait(1)

        # Create filter window rectangle
        window = SurroundingRectangle(
            VGroup(*[input_grid.get_cell(0+m, 0+n) for m in range(filter_size) for n in range(filter_size)]),
            color=YELLOW,
            buff=0.1
        )
        self.play(Create(window))
        self.wait(0.5)

        # Convolution Operation
        result_texts = {}
        for i in range(output_size):
            for j in range(output_size):
                # Move window
                new_cells = VGroup(*[input_grid.get_cell(i+m, j+n) for m in range(filter_size) for n in range(filter_size)])
                new_window = SurroundingRectangle(new_cells, color=YELLOW, buff=0.1)
                self.play(Transform(window, new_window), run_time=0.1)

                # Highlight cells
                highlights = []
                for m in range(filter_size):
                    for n in range(filter_size):
                        highlights.append(input_grid.get_cell(i+m, j+n))
                        highlights.append(filter_grid.get_cell(m, n))
                self.play(*[cell.animate.set_fill(YELLOW, opacity=0.5) for cell in highlights], run_time=0.1)

                # Build correct full operation text
                operation_parts = []
                total_sum = 0
                for m in range(filter_size):
                    for n in range(filter_size):
                        val_in = int(image_values[i+m, j+n])
                        val_filt = int(filter_values[m, n])
                        mult_result = val_in * val_filt
                        total_sum += mult_result
                        operation_parts.append(f"({val_in} \\times {val_filt})")

                # Join parts correctly
                full_operation = " + ".join(operation_parts) + f" = {total_sum}"

                # If very long, split into 2 lines
                if len(full_operation) > 80:
                    # crude split into two halves
                    mid = len(operation_parts) // 2
                    upper = " + ".join(operation_parts[:mid])
                    lower = " + ".join(operation_parts[mid:])
                    operation_text = MathTex(upper + r"\\" + lower + f"= {total_sum}", font_size=30)
                else:
                    operation_text = MathTex(full_operation, font_size=32)

                operation_text.move_to(DOWN * 2.5)

                self.play(FadeIn(operation_text), run_time=0.1)

                # Show the result
                result = Text(f"{total_sum:.1f}", font_size=20)
                result.move_to(output_grid.get_cell_center(i, j))
                self.play(FadeIn(result), run_time=0.1)
                result_texts[(i, j)] = result

                # Clear operation
                self.play(FadeOut(operation_text), run_time=0.1)

                # Unhighlight
                self.play(*[cell.animate.set_fill(opacity=0) for cell in highlights], run_time=0.1)

        # Clean up
        self.play(FadeOut(window))
        self.wait(0.5)

        # Final Celebration
        all_results = VGroup(*result_texts.values())
        self.play(all_results.animate.set_color(GREEN), run_time=0.1)

    def create_grid(self, size, values, scale=1, color=WHITE):
        grid = VGroup()
        for i in range(size):
            for j in range(size):
                square = Square(side_length=scale)
                square.set_stroke(color, 2)
                square.move_to(np.array([j * scale, -i * scale, 0]))
                number = Text(f"{values[i, j]:.0f}", font_size=24)
                number.move_to(square.get_center())
                grid.add(VGroup(square, number))
        grid.get_cell = lambda i, j: grid[i * size + j][0]
        grid.get_cell_center = lambda i, j: grid[i * size + j][0].get_center()
        return grid

    def create_empty_grid(self, size, scale=1, color=WHITE):
        grid = VGroup()
        for i in range(size):
            for j in range(size):
                square = Square(side_length=scale)
                square.set_stroke(color, 2)
                square.move_to(np.array([j * scale, -i * scale, 0]))
                grid.add(VGroup(square))
        grid.get_cell = lambda i, j: grid[i * size + j][0]
        grid.get_cell_center = lambda i, j: grid[i * size + j][0].get_center()
        return grid
