# # from manim import *

# # # manim -pql layer_norm.py LayerNormVisualization

from manim import *

class LayerNormVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Layer Normalization (Per Sample)").scale(0.7).to_edge(UP)
        self.play(Write(title))

        # Batch info
        batch_text = Text("4 Samples, 3 Features (R, G, B) each", font_size=30).next_to(title, DOWN)
        self.play(FadeIn(batch_text))
        
        # FIRST: Show the samples representation (4 individual samples, each with RGB features)
        sample_spacing = 1.0
        
        # Create 4 samples with RGB features
        samples = VGroup()
        for i in range(4):
            # Create RGB features for each sample
            blue_feature = Square(side_length=1.5, color=BLUE, fill_opacity=1)
            green_feature = Square(side_length=1.5, color=GREEN, fill_opacity=1).shift(UP*0.1 + LEFT*0.1)
            red_feature = Square(side_length=1.5, color=RED, fill_opacity=1).shift(UP*0.2 + LEFT*0.2)
            
            # Group the features for this sample
            sample = VGroup(blue_feature, green_feature, red_feature)
            
            # Position the sample
            sample_position = LEFT*4.5 + RIGHT*(i * (1.5 + sample_spacing))
            sample.move_to(sample_position)
            
            samples.add(sample)
        
        # Add dots to indicate more samples
        if i < 3:  # Only add dots after the third sample
            dots = Text("...", font_size=40).next_to(samples, RIGHT, buff=0.5)
            samples.add(dots)
        
        # Show all samples
        self.play(FadeIn(samples))
        
        # Create 4 explanation texts to be shown sequentially
        explanation_text1 = Text("Input data consists of multiple samples with different features", 
                              font_size=28).next_to(samples, DOWN, buff=0.7)
        explanation_text2 = Text("Each sample has its own distribution of feature values", 
                              font_size=28).next_to(samples, DOWN, buff=0.7)
        explanation_text3 = Text("Layer Norm normalizes each sample separately", 
                              font_size=28).next_to(samples, DOWN, buff=0.7)
        explanation_text4 = Text("This helps training by reducing internal covariate shift", 
                              font_size=28).next_to(samples, DOWN, buff=0.7)
        
        # Show explanations one after another with transitions
        self.play(Write(explanation_text1))
        self.wait(1.5)
        self.play(FadeOut(explanation_text1), FadeIn(explanation_text2))
        self.wait(1.5)
        self.play(FadeOut(explanation_text2), FadeIn(explanation_text3))
        self.wait(1.5)
        self.play(FadeOut(explanation_text3), FadeIn(explanation_text4))
        self.wait(1.5)
        
        # Remove samples and last text to make room for the next visualization
        self.play(FadeOut(samples), FadeOut(explanation_text4))
        
        # THEN: Continue with the visualization showing how Layer Norm works
        # Create 3 example "features" for Sample 1: R, G, B
        sample_label = Text("Sample 1", font_size=24).shift(2*UP + LEFT*4)
        self.play(Write(sample_label))
        
        r_blocks = VGroup(*[Square(color=RED, fill_opacity=1).scale(0.4) for _ in range(1)]).arrange(RIGHT, buff=0.2).shift(2*UP + RIGHT*2)
        g_blocks = VGroup(*[Square(color=GREEN, fill_opacity=1).scale(0.4) for _ in range(1)]).arrange(RIGHT, buff=0.2).shift(RIGHT*2)
        b_blocks = VGroup(*[Square(color=BLUE, fill_opacity=1).scale(0.4) for _ in range(1)]).arrange(RIGHT, buff=0.2).shift(2*DOWN + RIGHT*2)
        self.play(FadeOut(sample_label))
        self.play(FadeIn(r_blocks), FadeIn(g_blocks), FadeIn(b_blocks))

        # Slide them to the left
        self.play(
            r_blocks.animate.shift(LEFT*4),
            g_blocks.animate.shift(LEFT*4),
            b_blocks.animate.shift(LEFT*4),
            run_time=1.5
        )

        # Now add the labels
        r_label = Text("R Feature", font_size=24).next_to(r_blocks, LEFT)
        g_label = Text("G Feature", font_size=24).next_to(g_blocks, LEFT)
        b_label = Text("B Feature", font_size=24).next_to(b_blocks, LEFT)

        self.play(Write(r_label), Write(g_label), Write(b_label))

        # Show equations — positioned to the right of the feature blocks
        equations_group = VGroup(
            MathTex(r"\text{For each sample compute:}", font_size=30),
            MathTex(r"\mu_n = \frac{1}{C \cdot H \cdot W} \sum x_{n,c,h,w}", font_size=30),
            MathTex(r"\sigma^2_n = \frac{1}{C \cdot H \cdot W} \sum (x_{n,c,h,w} - \mu_n)^2", font_size=30),
            MathTex(r"\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_n}{\sqrt{\sigma^2_n + \epsilon}}", font_size=30),
            MathTex(r"y_{n,c,h,w} = \gamma \, \hat{x}_{n,c,h,w} + \beta", font_size=30)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(g_blocks, RIGHT, buff=1.5)

        self.play(Write(equations_group[0]))
        self.wait(0.3)
        self.play(Write(equations_group[1]))
        self.wait(0.3)
        self.play(Write(equations_group[2]))
        self.wait(0.3)
        self.play(Write(equations_group[3]))
        self.wait(0.3)
        self.play(Write(equations_group[4]))
        self.wait(2)
        
        # Clear screen for numerical example
        self.play(
            FadeOut(r_blocks), FadeOut(g_blocks), FadeOut(b_blocks),
            FadeOut(r_label), FadeOut(g_label), FadeOut(b_label),
            FadeOut(equations_group), FadeOut(sample_label),
            FadeOut(title), FadeOut(batch_text)
        )
        
        # Add numerical example title
        example_title = Text("Layer Normalization: Numerical Example").scale(0.8).to_edge(UP)
        self.play(Write(example_title))
        
        # Show numerical example for just one sample
        example_text = Text("Sample 1 Example (with R, G, B features)", font_size=28).next_to(example_title, DOWN)
        self.play(Write(example_text))
        
        # Create a table for original values
        original_values = [125, 240, 35]  # Example values for R, G, B features
        
        # Original values table
        original_table = Table(
            [["R Feature", "G Feature", "B Feature"],
             [str(val) for val in original_values]],
            row_labels=[Text("Feature"), Text("Value")],
            include_outer_lines=True
        ).scale(0.6).shift(UP*1)
        
        self.play(Create(original_table))
        
        # Step 1: Calculate mean
        mean_value = sum(original_values) / len(original_values)
        mean_eq = MathTex(
            r"\mu_n = \frac{1}{C} \sum x_{n,c} = \frac{125 + 240 + 35}{3} = ", f"{mean_value:.1f}"
        ).scale(0.8).next_to(original_table, DOWN, buff=0.5)
        
        self.play(Write(mean_eq))
        self.wait(1)
        
        # Step 2: Calculate variance
        var_values = [(val - mean_value)**2 for val in original_values]
        variance = sum(var_values) / len(var_values)
        
        var_eq = MathTex(
            r"\sigma^2_n = \frac{1}{C} \sum (x_{n,c} - \mu_n)^2 = ", f"{variance:.1f}"
        ).scale(0.8).next_to(mean_eq, DOWN, buff=0.3)
        
        self.play(Write(var_eq))
        self.wait(1)
        
        # Step 3: Normalize (epsilon = 1e-5 for numerical stability)
        epsilon = 1e-5
        normalized_values = [(val - mean_value) / (variance + epsilon)**0.5 for val in original_values]
        
        norm_eq = MathTex(
            r"\hat{x}_{n,c} = \frac{x_{n,c} - \mu_n}{\sqrt{\sigma^2_n + \epsilon}}"
        ).scale(0.8).next_to(var_eq, DOWN, buff=0.3)
        
        self.play(Write(norm_eq))
        self.wait(1)
        
        # Clear screen before showing normalized table
        self.play(
            FadeOut(original_table),
            FadeOut(mean_eq),
            FadeOut(var_eq),
            FadeOut(norm_eq),
            FadeOut(example_title),
            FadeOut(example_text)
        )

        # New screen: Title for normalized values
        norm_screen_title = Text("Normalized Feature Values for Sample 1").scale(0.8).to_edge(UP)
        self.play(Write(norm_screen_title))
        
        # Show the normalized values in a table
        norm_table = Table(
            [["R Feature", "G Feature", "B Feature"],
             [f"{val:.2f}" for val in normalized_values]],
            row_labels=[Text("Feature"), Text("Normalized")],
            include_outer_lines=True
        ).scale(0.6).next_to(norm_screen_title, DOWN, buff=0.5)
        
        self.play(Create(norm_table))
        self.wait(1)
        
        # Step 4: Scale and shift (γ = 1.5, β = 100)
        gamma = 1.5
        beta = 100
        scaled_values = [gamma * val + beta for val in normalized_values]
        
        scale_eq = MathTex(
            r"y_{n,c} = \gamma \cdot \hat{x}_{n,c} + \beta = ", f"{gamma:.1f}", r" \cdot \hat{x}_{n,c} + ", f"{beta}"
        ).scale(0.8).next_to(norm_table, DOWN, buff=0.5)
        
        self.play(Write(scale_eq))
        self.wait(1)
        
        # Final values table
        final_table = Table(
            [["R Feature", "G Feature", "B Feature"],
             [f"{val:.1f}" for val in scaled_values]],
            row_labels=[Text("Feature"), Text("Output")],
            include_outer_lines=True
        ).scale(0.6).next_to(scale_eq, DOWN, buff=0.5)
        
        self.play(Create(final_table))
        
        # Final explanation
        final_text = Text("γ and β are learnable parameters that restore representation power", 
                        font_size=24).next_to(final_table, DOWN, buff=0.5)
        self.play(Write(final_text))
        
        # Clear previous screen
        self.play(
            FadeOut(norm_screen_title),
            FadeOut(norm_table),
            FadeOut(final_table),
            FadeOut(scale_eq),
            FadeOut(final_text)
        )

        # New screen: What's different about Layer Norm vs Batch Norm?
        headline = Text("🧪 Layer Normalization vs Batch Normalization", font_size=30).to_edge(UP)
        self.play(Write(headline))

        # Key Differences
        difference_title = Text("🔑 Key Differences:", font_size=28, color=RED).next_to(headline, DOWN, aligned_edge=LEFT).shift(DOWN*0.5)
        difference_points = VGroup(
            Text("• Layer Norm: normalizes across features within each sample", font_size=24),
            Text("• Batch Norm: normalizes across samples for each feature", font_size=24),
            Text("• Layer Norm works well with variable batch sizes", font_size=24),
            Text("• Layer Norm is independent of batch statistics", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(difference_title, DOWN, aligned_edge=LEFT)

        self.play(Write(difference_title), FadeIn(difference_points, shift=RIGHT))
        self.wait(2)

        # Advantages Section
        advantages_title = Text("✅ Advantages of Layer Normalization:", font_size=28, color=RED).next_to(difference_points, DOWN, aligned_edge=LEFT).shift(DOWN*0.7)
        advantages_points = VGroup(
            Text("• Works well with small batch sizes", font_size=24),
            Text("• Effective for RNNs and Transformers", font_size=24),
            Text("• Same computation during training and inference", font_size=24),
            Text("• No need to track running statistics", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(advantages_title, DOWN, aligned_edge=LEFT)

        self.play(Write(advantages_title), FadeIn(advantages_points, shift=RIGHT))
        self.wait(2)
        # Clear previous screen
        self.play(
            FadeOut(difference_title),
            FadeOut(difference_points),
            FadeOut(advantages_title),
            FadeOut(advantages_points),
            FadeOut(headline)
        )
        
        # Title for comparison table
        comparison_title = Text("📋 Layer Normalization vs Batch Normalization").scale(0.7).to_edge(UP)
        self.play(Write(comparison_title))

        # Define table data
        comparison_features = [
            ["Feature", "Layer Normalization", "Batch Normalization"],
            ["Where it normalizes", "Across features (per sample)", "Across batch (per feature)"],
            ["Formula", "Normalizes all features per sample", "Normalizes each feature across samples"],
            ["Best for", "RNNs, Transformers", "CNNs, deep feedforward nets"],
            ["Batch size dependency", "❌ No – works with any batch size", "✅ Yes – needs large batch size"],
            ["During inference", "Same computation as training", "Uses running averages"],
            ["Learnable parameters", "✅ Yes (γ, β per layer)", "✅ Yes (γ, β per feature)"],
            ["Statistical independence", "✅ Yes (between samples)", "❌ No (samples affect each other)"],
        ]

        # Create table
        comparison_table = Table(
            [row for row in comparison_features[1:]],  # exclude header for content
            col_labels=[Text(comparison_features[0][0], font_size=26), 
                        Text(comparison_features[0][1], font_size=26),
                        Text(comparison_features[0][2], font_size=26)],
            element_to_mobject=lambda s: Text(s, font_size=26, line_spacing=0.8),
            include_outer_lines=True
        ).scale(0.65).next_to(comparison_title, DOWN, buff=0.4)

        self.play(Create(comparison_table))
        self.wait(4)
        
        self.wait(3)