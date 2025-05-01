from manim import *

# manim -pql Manim/skip_conn.py SkipConnectionResNet

from manim import *

class SkipConnectionResNet(Scene):
    def construct(self):
        # Setup input, conv, addition blocks
        input_block = Rectangle(width=2, height=1, color=BLUE).shift(LEFT*4)
        input_text = Text("Input", font_size=24).move_to(input_block)

        conv_block = Rectangle(width=2, height=1, color=GREEN).shift(ORIGIN)
        conv_text = Text("Conv Block", font_size=24).move_to(conv_block)

        add_block = Rectangle(width=2, height=1, color=YELLOW).shift(RIGHT*4)
        add_text = Text("Add", font_size=24).move_to(add_block)

        arrow1 = Arrow(input_block.get_right(), conv_block.get_left(), buff=0.1)
        arrow2 = Arrow(conv_block.get_right(), add_block.get_left(), buff=0.1)

        skip_curve = ArcBetweenPoints(
            input_block.get_top(),
            add_block.get_top(),
            angle=-PI/2,
            color=RED
        )

        arrow_tip = Triangle(color=RED, fill_opacity=1).scale(0.1)
        arrow_tip.move_to(skip_curve.get_end())
        tangent_angle = skip_curve.get_angle()
        arrow_tip.rotate(tangent_angle)

        skip_arrow = VGroup(skip_curve, arrow_tip)
        skip_label = Text("Skip", font_size=20, color=RED).next_to(skip_curve, UP)

        input_group = VGroup(input_block, input_text)
        conv_group = VGroup(conv_block, conv_text)
        add_group = VGroup(add_block, add_text)

        explanation = Text("", font_size=28).to_edge(DOWN)

        # --- Start Animation ---
        self.play(FadeIn(input_group))
        self.play(FadeIn(explanation))
        self.play(Transform(explanation, Text("Starting with Input", font_size=28).to_edge(DOWN)))
        self.wait(1)

        self.play(GrowArrow(arrow1))
        self.play(FadeIn(conv_group))
        self.play(Transform(explanation, Text("Passing through Conv Block", font_size=28).to_edge(DOWN)))
        self.wait(1)

        self.play(GrowArrow(arrow2))
        self.play(FadeIn(add_group))
        self.play(Transform(explanation, Text("Conv output goes to Addition", font_size=28).to_edge(DOWN)))
        self.wait(1)

        self.play(Create(skip_curve), FadeIn(arrow_tip), FadeIn(skip_label))
        self.play(Transform(explanation, Text("Skip connection sends input directly", font_size=28).to_edge(DOWN)))
        self.wait(2)

        self.play(add_block.animate.set_fill(YELLOW, opacity=0.5))
        self.play(Indicate(add_text))
        self.play(Transform(explanation, Text("Addition merges both paths", font_size=28).to_edge(DOWN)))
        self.wait(2)

        # --- Transition to next section ---
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

        # --- Frame 2: Advantages ---
        advantages_title = Text("✅ Advantages of Skip Connections", font_size=40, color=GREEN).to_edge(UP)
        advantages = BulletedList(
            "Prevent vanishing gradients",
            "Enable deeper networks",
            "Improve learning speed",
            "Better accuracy",
            font_size=30
        ).shift(DOWN*0.5)

        self.play(FadeIn(advantages_title))
        self.play(FadeIn(advantages, lag_ratio=0.3, run_time=3))
        self.wait(3)

        # --- Transition ---
        self.play(FadeOut(advantages_title), FadeOut(advantages))
        self.wait(0.5)

        # --- Frame 3: When to use ---
        when_to_use_title = Text("📌 When to Use Skip Connections", font_size=40, color=BLUE).to_edge(UP)
        when_to_use = BulletedList(
            "Very deep neural networks (e.g., ResNet18, ResNet50)",
            "Training becomes unstable",
            "You observe vanishing gradients",
            font_size=30
        ).shift(DOWN*0.5)

        self.play(FadeIn(when_to_use_title))
        self.play(FadeIn(when_to_use, lag_ratio=0.3, run_time=3))
        self.wait(3)

        # --- Transition ---
        self.play(FadeOut(when_to_use_title), FadeOut(when_to_use))
        self.wait(0.5)

        # --- Frame 4: When NOT to use ---
        when_not_use_title = Text("⚠️ When NOT to Use Skip Connections", font_size=40, color=RED).to_edge(UP)
        when_not_use = BulletedList(
            "Small, shallow models",
            "Tasks where simpler models are enough",
            "Skip connections might cause overfitting",
            font_size=30
        ).shift(DOWN*0.5)

        self.play(FadeIn(when_not_use_title))
        self.play(FadeIn(when_not_use, lag_ratio=0.3, run_time=3))
        self.wait(3)

        # --- End ---
        self.play(FadeOut(when_not_use_title), FadeOut(when_not_use))
        ending = Text("Thanks for Watching! 🎬", font_size=36).move_to(ORIGIN)
        self.play(FadeIn(ending))
        self.wait(3)
