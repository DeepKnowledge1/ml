from manim import *
import numpy as np

# manim -pql transfer_learning.py TransferLearningAnimation

class TransferLearningAnimation(Scene):
    def construct(self):
        # Title and introduction
        title = Text("Transfer Learning", font_size=48)
        subtitle = Text("Leveraging pre-trained knowledge for new tasks", font_size=28)
        subtitle.next_to(title, DOWN, buff=0.5)
        title_group = VGroup(title, subtitle)
        title_group.to_edge(UP, buff=1.2)  # Increased buffer for more space

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(1)
        self.play(title_group.animate.scale(0.6).to_edge(UP, buff=0.5))  # Smaller scale, less buffer space

        # Helper function for neural networks
        def create_neural_network(layers_sizes, position=ORIGIN, color=BLUE):
            nn = VGroup()
            layers = []
            for i, layer_size in enumerate(layers_sizes):
                layer = VGroup()
                vertical_spacing = min(0.6, 2.0 / layer_size)
                vertical_offset = (layer_size - 1) * vertical_spacing / 2
                for j in range(layer_size):
                    neuron = Circle(radius=0.1, color=color, fill_opacity=0.5)
                    neuron.move_to([i * 1.5, j * vertical_spacing - vertical_offset, 0])
                    layer.add(neuron)
                layers.append(layer)
                nn.add(layer)
            connections = VGroup()
            for i in range(len(layers) - 1):
                for n1 in layers[i]:
                    for n2 in layers[i + 1]:
                        connection = Line(n1.get_center(), n2.get_center(), stroke_opacity=0.3, stroke_width=0.8)
                        connections.add(connection)
            nn.add(connections)
            nn.move_to(position)
            return nn, layers
        source_nn, source_layers = create_neural_network([5, 8, 8, 5], position=LEFT * 4, color=BLUE)

        # Source model
        source_title = Text("Source Model (Pre-trained)", font_size=28, color=BLUE)
        source_title.to_edge(UP, buff=1.8)  # More buffer space
        source_title.shift(LEFT * 4)  # Reduced shift to prevent overlap
        source_nn, source_layers = create_neural_network([5, 8, 8, 5], position=LEFT * 4, color=BLUE)
        source_input_label = Text("Images", font_size=18).next_to(source_nn, LEFT, buff=0.5)
        source_output_label = Text("Classes", font_size=18).next_to(source_nn, RIGHT, buff=0.5)

        self.play(Write(source_title), run_time=1)
        self.play(FadeIn(source_nn), run_time=1.5)
        self.play(Write(source_input_label), Write(source_output_label), run_time=1)
        self.wait(1)

        # Training animation - more dynamic pulsing effect
        for i in range(2):
            self.play(
                *[source_nn[j][k].animate.set_fill(BLUE, opacity=0.8) 
                  for j in range(len(source_layers)) for k in range(len(source_layers[j]))],
                run_time=1.5,
                rate_func=there_and_back
            )

        # Learned features - positioned better
        learned_features = Text("Learned Features", font_size=22, color=YELLOW)
        learned_features.next_to(source_nn, DOWN, buff=0.6)
        arrow = Arrow(learned_features.get_top(), source_nn.get_bottom(), color=YELLOW, buff=0.2)
        self.play(Create(arrow), Write(learned_features), run_time=1.5)
        self.wait(1)

        # Target model - adjusted positioning
        target_title = Text("Target Model (New Task)", font_size=28, color=GREEN)
        target_title.to_edge(UP, buff=1.8)  # More buffer space
        target_title.shift(RIGHT * 4)  # Reduced shift to prevent overlap
        target_nn, target_layers = create_neural_network([5, 8, 8, 2], position=RIGHT * 4, color=GREEN_D)
        target_input_label = Text("Medical Images", font_size=18).next_to(target_nn, LEFT, buff=0.5)
        target_output_label = Text("Diagnosis", font_size=18).next_to(target_nn, RIGHT, buff=0.5)

        self.play(Write(target_title), run_time=1)
        self.play(FadeIn(target_nn), FadeIn(target_input_label), FadeIn(target_output_label), run_time=1.5)
        self.wait(1)

        # Transfer arrow - adjusted for better positioning
        transfer_arrow = CurvedArrow(
            source_nn.get_right() + UP * 0.5, 
            target_nn.get_left() + UP * 0.5, 
            color=YELLOW, 
            angle=0.3
        )
        transfer_text = Text("Transfer", font_size=22, color=YELLOW).move_to(
            transfer_arrow.point_from_proportion(0.5) + UP * 0.4
        )
        self.play(Create(transfer_arrow), Write(transfer_text), run_time=1.5)
        self.wait(1)

        # Frozen/trainable layers - clearer visual distinction
        frozen_rect = Rectangle(
            width=target_nn.get_width() * 0.5, 
            height=target_nn.get_height(),
            stroke_color=BLUE,
            stroke_width=2,
            fill_color=BLUE,
            fill_opacity=0.05
        ).align_to(target_nn, LEFT).shift(LEFT * 0.1)
        
        trainable_rect = Rectangle(
            width=target_nn.get_width() * 0.5,
            height=target_nn.get_height(),
            stroke_color=GREEN,
            stroke_width=2,
            fill_color=GREEN,
            fill_opacity=0.05
        ).align_to(target_nn, RIGHT).shift(RIGHT * 0.1)
        
        freeze_text = Text("Frozen Layers", font_size=16, color=BLUE).next_to(frozen_rect, UP, buff=0.2)
        trainable_text = Text("Trainable Layers", font_size=16, color=GREEN).next_to(trainable_rect, UP, buff=0.2)

        self.play(
            Create(frozen_rect), 
            Create(trainable_rect),
            Write(freeze_text), 
            Write(trainable_text), 
            run_time=1.5
        )
        self.wait(1)

        # Highlight frozen layers - more distinctive animation
        for j in range(2):
            pulse_anims = []
            for i in range(2):  # First two layers are frozen
                for k in range(len(target_layers[i])):
                    pulse_anims.append(
                        target_layers[i][k].animate.set_fill(BLUE, opacity=0.8)
                    )
            self.play(
                *pulse_anims,
                run_time=0.8,
                rate_func=there_and_back
            )

        # Highlight trainable layers - more distinctive animation
        for j in range(2):
            pulse_anims = []
            for i in range(2, 4):  # Last two layers are trainable
                for k in range(len(target_layers[i])):
                    pulse_anims.append(
                        target_layers[i][k].animate.set_fill(GREEN, opacity=0.8)
                    )
            self.play(
                *pulse_anims,
                run_time=0.8,
                rate_func=there_and_back
            )

        self.wait(1)

        # Fine-tuning label - better positioned
        finetune_text = Text("Fine-Tuning", font_size=28, color=YELLOW).next_to(target_nn, DOWN, buff=0.6)
        finetune_arrow = Arrow(finetune_text.get_top(), trainable_rect.get_bottom(), color=YELLOW, buff=0.2)
        
        self.play(Write(finetune_text), Create(finetune_arrow), run_time=1)
        
        # Fine-tuning animation - more dynamic
        for _ in range(2):
            self.play(
                *[target_layers[i][j].animate.set_fill(GREEN, opacity=0.9) 
                  for i in range(2, 4) for j in range(len(target_layers[i]))],
                rate_func=there_and_back_with_pause,
                run_time=1.5
            )
        self.wait(1)

        # Benefits - ensure proper positioning with no overlap
        self.play(
            FadeOut(freeze_text),
            FadeOut(trainable_text),
            FadeOut(frozen_rect),
            FadeOut(trainable_rect),
            FadeOut(finetune_arrow),
            run_time=0.8
        )
        

        # Benefits - New clean window
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1
        )

        self.wait(0.5)
        
        benefits_title = Text("Benefits of Transfer Learning", font_size=32, color=YELLOW)
        benefits_title.to_edge(DOWN, buff=2.0)
        
        benefit1 = Text("• Less training data needed", font_size=24)
        benefit2 = Text("• Faster training time", font_size=24)
        benefit3 = Text("• Better performance", font_size=24)
        
        benefits = VGroup(benefit1, benefit2, benefit3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        benefits.next_to(benefits_title, DOWN, buff=0.4)
        
        # Ensure benefits stay within frame
        benefits_group = VGroup(benefits_title, benefits)
        if benefits.get_bottom()[1] < -3.5:
            benefits_group.shift(UP * (abs(benefits.get_bottom()[1]) - 3))

        self.play(Write(benefits_title), run_time=1)
        
        # Animate benefits appearing one by one
        for benefit in [benefit1, benefit2, benefit3]:
            self.play(Write(benefit), run_time=0.7)
        
        self.wait(2)

        # Final fade out - more organized
        self.play(
            FadeOut(VGroup(
                source_nn, target_nn, source_input_label, source_output_label,
                target_input_label, target_output_label, source_title, target_title,
                transfer_arrow, transfer_text, finetune_text, learned_features,
                arrow, benefits_group
            )),
            run_time=1.5
        )

        # Final screen - more dynamic and catchy
        # Get dimensions of the frame
        frame_width = config.frame_width
        frame_height = config.frame_height
        
        final_bg = Rectangle(
            width=frame_width,
            height=frame_height,
            fill_color=BLACK,
            fill_opacity=1,
            stroke_width=0
        )
        
        final_title = Text("Transfer Learning", font_size=64, color=BLUE_A).to_edge(UP, buff=1.5)
        final_subtitle = Text("Efficient learning for the AI era", font_size=36, color=GREEN_A)
        final_subtitle.next_to(final_title, DOWN, buff=0.8)
        
        # Create a simple neural network icon
        nn_icon = VGroup()
        for i in range(3):
            for j in range(3):
                if i == 1 or j == 1:  # Create a plus shape
                    dot = Dot(point=[i-1, j-1, 0], radius=0.1, color=YELLOW_A)
                    nn_icon.add(dot)
        
        for i in range(5):
            for j in range(2):
                connection = Line(
                    [-0.5 + j, -1 + i*0.5, 0],
                    [0.5 + j, -1 + i*0.5, 0],
                    stroke_width=2,
                    stroke_opacity=0.7,
                    color=BLUE_A
                )
                nn_icon.add(connection)
        
        nn_icon.scale(1.5).next_to(final_subtitle, DOWN, buff=1.0)
        
        self.play(
            FadeIn(final_bg),
            Write(final_title),
            run_time=1.5
        )
        self.play(
            Write(final_subtitle),
            FadeIn(nn_icon),
            run_time=1.5
        )
        
        # Add a final animation to nn_icon to keep it engaging
        self.play(
            nn_icon.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(2)