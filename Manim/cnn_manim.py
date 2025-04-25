from manim import *
import random
from manim.utils.color import WHITE, BLACK
from manim.utils.tex_templates import TexFontTemplates


# manim -pql cnn_manim.py DropoutUsageTable

# manim -pqh cnn_manim.py DropoutInCNN DropoutUsageTable DropoutComparison DropoutCodeCompare


from manim import *
import random

class DropoutInCNN(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Title
        title = Text("Dropout in CNNs", font_size=56, weight=BOLD).set_color(BLACK).to_edge(UP)
        underline = Line(start=LEFT * 3.5, end=RIGHT * 3.5, color=BLACK).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), Create(underline))
        self.wait(1)

        # Grids
        input_map = self.create_grid(3, 3, BLUE_D).scale(1.2).shift(LEFT * 4 + DOWN)
        conv_map = self.create_grid(3, 3, GREEN_D).scale(1.2).shift(RIGHT * 0.5 + DOWN)
        dropout_map = self.create_grid(3, 3, RED_D).scale(1.2).shift(RIGHT * 5 + DOWN)

        # Labels
        input_label = Text("Input Feature Map", font_size=28).next_to(input_map, DOWN, buff=0.3).set_color(BLACK)
        conv_label = Text("Conv Layer Output", font_size=28).next_to(conv_map, DOWN, buff=0.3).set_color(BLACK)
        dropout_label = Text("After Dropout", font_size=28).next_to(dropout_map, DOWN, buff=0.3).set_color(BLACK)

        self.play(FadeIn(input_map), FadeIn(conv_map), FadeIn(dropout_map))
        self.play(Write(input_label), Write(conv_label), Write(dropout_label))

        # Arrows
        arrow1 = Arrow(input_map.get_right(), conv_map.get_left(), buff=0.2, color=BLACK)
        arrow2 = Arrow(conv_map.get_right(), dropout_map.get_left(), buff=0.2, color=BLACK)
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))

        # Explanation text
        explanation = Text(
            "Dropout randomly zeroes out some activations during training.",
            font_size=26, slant=ITALIC
        ).next_to(title, DOWN, buff=1).set_color(GRAY_E)
        self.play(Write(explanation))

        # Animate dropout
        squares = dropout_map.submobjects
        to_zero = random.sample(squares, k=4)
        for s in to_zero:
            s.generate_target()
            s.target.set_fill(GRAY_B, opacity=0.15)
            s.target.set_stroke(GRAY_D, width=1.5)
        self.play(*[MoveToTarget(s) for s in to_zero])

        # Final note
        note = Text("Dropout helps prevent overfitting and improves generalization.",
                    font_size=24, color=GRAY_E).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(note))
        self.wait(3)
        

    def create_grid(self, rows, cols, color=BLUE, size=0.7):
        grid = VGroup()
        for i in range(rows):
            for j in range(cols):
                square = Square(side_length=size)
                square.set_stroke(color=color, width=2)
                square.set_fill(color=color, opacity=0.3)
                square.move_to(np.array([j - cols / 2 + 0.5, rows / 2 - i - 0.5, 0]) * size * 1.3)
                grid.add(square)
        return grid


        

class DropoutQuickTips(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        
        # Title
        title = Text("Quick Rule of Thumb", font_size=40, weight=BOLD, color=BLACK).to_edge(UP)
        underline = Line(LEFT * 4, RIGHT * 4, color=BLACK).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), Create(underline))
        self.wait(2)
        
        # Table with strings including emojis
        headers = ["Situation", "Use Dropout?"]
        rows = [
            ["Small dataset", "✅ Yes"],
            ["Large dataset", "🚫 Probably not"],
            ["Overfitting observed", "✅ Definitely"],
            ["Underfitting observed", "🚫 No"],
            ["Using BatchNorm", "🤔 Be cautious"],
            ["Using Transformers", "✅ But with care"],
        ]
        
        # Create table with strings
        table = Table(
            rows,
            col_labels=[Text(h, font_size=70, color=BLACK) for h in headers],
            include_outer_lines=True,
            line_config={"stroke_color": BLACK}
        ).scale(0.5).next_to(underline, DOWN, buff=1)
        
        # Set text color and font for all entries to ensure emoji support
        for mob in table.get_entries():
            mob.set_color(BLACK)
            # Try a different font that might support emojis better
            # You might need to experiment with different fonts available on your system
            if hasattr(mob, "font"):  # Check if the font attribute exists
                mob.font = "Arial"
        
        self.play(FadeIn(table))
        self.wait(4)        
        
        
from manim import *

class DropoutComparison(Scene):
    def construct(self):
        title = Text("Dropout: Training vs Inference", font_size=48, weight=BOLD)
        title.to_edge(UP)

        # Training Column
        training_title = Text("Training", color=BLUE, font_size=36)
        training_text = VGroup(
            Text("✓ Dropout is ON", font_size=30),
            Text("✓ Random neurons are dropped", font_size=30),
            Text("✓ Promotes generalization", font_size=30),
            Text("✓ Use: model.train()", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        training = VGroup(training_title, training_text).arrange(DOWN).to_edge(LEFT).shift(RIGHT * 0.5)

        # Inference Column
        inference_title = Text("Inference", color=GREEN, font_size=36)
        inference_text = VGroup(
            Text("✗ Dropout is OFF", font_size=30),
            Text("✗ All neurons are used", font_size=30),
            Text("✓ Output scaled (automatically)", font_size=30),
            Text("✓ Use: model.eval()", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        inference = VGroup(inference_title, inference_text).arrange(DOWN).to_edge(RIGHT).shift(LEFT * 0.5)

        self.play(Write(title))
        self.wait(1)

        self.play(FadeIn(training, shift=LEFT), FadeIn(inference, shift=RIGHT))
        self.wait(2)

        self.play(training.animate.shift(LEFT * 0.5), inference.animate.shift(RIGHT * 0.5))
        self.wait(3)





class DropoutUsageTable(Scene):
    def construct(self):
        title = Text("🧠 When to Use Dropout", font_size=48, weight=BOLD)
        title.to_edge(UP)

        # Table rows
        headers = ["Condition", "Use Dropout?"]
        data = [
            ["Small dataset", "✅ Yes"],
            ["Large dataset", "🚫 Probably not"],
            ["Overfitting observed", "✅ Definitely"],
            ["Underfitting observed", "🚫 No"],
            ["Using BatchNorm", "🤔 Be cautious"],
            ["Using Transformers", "✅ But with care"],
        ]

        # Create Table
        table = Table(
            [[*row] for row in data],
            col_labels=[Text(headers[0], font_size=32, weight=BOLD),
                        Text(headers[1], font_size=32, weight=BOLD)],
            top_left_entry=Text("🔍", font_size=28),
            include_outer_lines=True,
            line_config={"stroke_width": 2},
            element_to_mobject=lambda x: Text(x, font_size=28),
            h_buff=1.8,
            v_buff=0.5,
        )

        table.scale(0.8)
        table.next_to(title, DOWN, buff=0.6)

        self.play(Write(title))
        self.wait(1)
        self.play(Create(table))
        self.wait(3)
        
from manim import *

class DropoutCodeCompare(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Title
        title = Text("Dropout in NN vs CNN", font_size=52, weight=BOLD, color=BLACK).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # NN Code
        nn_code = Code(code="""nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)""", language="Python", font_size=30, background="window", style="monokai")

        # CNN Code
        cnn_code = Code(code="""nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.Dropout2d(0.25),
    nn.Flatten(),
    nn.Linear(32*28*28, 10)
)""", language="Python", font_size=30, background="window", style="monokai")

        nn_code.scale(0.6).to_edge(LEFT).shift(DOWN * 0.3)
        cnn_code.scale(0.6).to_edge(RIGHT).shift(DOWN * 0.3)

        # Labels
        nn_label = Text("Fully Connected NN", font_size=28, color=BLUE).next_to(nn_code, UP, buff=0.3)
        cnn_label = Text("Convolutional NN", font_size=28, color=GREEN).next_to(cnn_code, UP, buff=0.3)

        self.play(FadeIn(nn_label), FadeIn(nn_code), FadeIn(cnn_label), FadeIn(cnn_code))
        self.wait(5)

# class DropoutCNN(Scene):
#     def construct(self):
#         self.camera.background_color = WHITE

#         # Title
#         title = Text("Dropout in CNNs", font_size=56, weight=BOLD).set_color(BLACK).to_edge(UP)
#         underline = Line(start=LEFT * 3.5, end=RIGHT * 3.5, color=BLACK).next_to(title, DOWN, buff=0.2)
#         self.play(Write(title), Create(underline))
#         self.wait(0.5)

#         # Create input feature map
#         input_map = self.create_grid(3, 3, color=BLUE_D).scale(1.2).shift(LEFT * 4 + DOWN * 0.5)
#         input_label = Text("Input Feature Map", font_size=28).next_to(input_map, DOWN, buff=0.3).set_color(BLACK)
#         self.play(FadeIn(input_map), Write(input_label))

#         # Convolutional output
#         conv_map = self.create_grid(3, 3, color=GREEN_D).scale(1.2).shift(RIGHT * 0.5 + DOWN * 0.5)
#         conv_label = Text("Conv Layer Output", font_size=28).next_to(conv_map, DOWN, buff=0.3).set_color(BLACK)
#         self.play(FadeIn(conv_map), Write(conv_label))

#         # Dropout output
#         dropout_map = self.create_grid(3, 3, color=RED_D).scale(1.2).shift(RIGHT * 5 + DOWN * 0.5)
#         dropout_label = Text("After Dropout", font_size=28).next_to(dropout_map, DOWN, buff=0.3).set_color(BLACK)
#         self.play(FadeIn(dropout_map), Write(dropout_label))

#         # Arrows showing flow
#         arrow1 = Arrow(input_map.get_right(), conv_map.get_left(), buff=0.2, stroke_width=4, color=BLACK)
#         arrow2 = Arrow(conv_map.get_right(), dropout_map.get_left(), buff=0.2, stroke_width=4, color=BLACK)
#         self.play(GrowArrow(arrow1), GrowArrow(arrow2))
#         self.wait(0.5)

#         # Dropout explanation (move up!)
#         explanation = Text(
#             "Dropout randomly zeroes out some activations during training.",
#             font_size=26, slant=ITALIC
#         ).next_to(title, DOWN, buff=1).set_color(GRAY_E)
#         self.play(Write(explanation))

#         # Animate dropout effect (deactivate some activations)
#         squares = dropout_map.submobjects
#         num_to_zero = random.sample(squares, k=4)

#         for square in num_to_zero:
#             square.generate_target()
#             square.target.set_fill(color=GRAY_B, opacity=0.15)
#             square.target.set_stroke(color=GRAY_D, width=1.5)

#         self.wait(0.5)
#         self.play(*[MoveToTarget(sq) for sq in num_to_zero])
#         self.wait(1.5)

#         # Ending note, lower to bottom
#         end_note = Text(
#             "Dropout helps prevent overfitting and improves generalization.",
#             font_size=24
#         ).set_color(GRAY_E).to_edge(DOWN, buff=0.5)
#         self.play(FadeIn(end_note))
#         self.wait(3)

#     def create_grid(self, rows, cols, color=BLUE, size=0.7):
#         grid = VGroup()
#         for i in range(rows):
#             for j in range(cols):
#                 square = Square(side_length=size)
#                 square.set_stroke(color=color, width=2)
#                 square.set_fill(color=color, opacity=0.3)
#                 square.move_to(np.array([j - cols / 2 + 0.5, rows / 2 - i - 0.5, 0]) * size * 1.3)
#                 grid.add(square)
#         return grid
