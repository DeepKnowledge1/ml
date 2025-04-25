# # from manim import *

# # # manim -pql batchnorm.py BatchNormVisualization
# # # manim -pql batchnorm.py BatchNormVisualization
# # # manim -pql batchnorm.py BatchNormInteractive


# from manim import *

# class BatchNormVisualization(Scene):
#     def construct(self):
#         # Title
#         title = Text("Batch Normalization (Per Channel)").scale(0.7).to_edge(UP)
#         self.play(Write(title))

#         # Batch info
#         batch_text= MathTex(r"4 \text{ Batches}, 3 \text{ Channels (R, G, B),  32x32 each}", font_size=30).next_to(title, DOWN)
#         # batch_text = Text("4 Batches, 3 Channels (R, G, B), 32×32 each", font_size=30).next_to(title, DOWN)
#         self.play(FadeIn(batch_text))
        
#         # FIRST: Show the batches representation (4 groups of RGB channels)
#         batch_spacing = 1.0
        
#         # Create 4 batches with RGB channels
#         batches = VGroup()
#         for i in range(4):
#             # Create RGB layers for each batch
#             blue_layer = Square(side_length=1.5, color=BLUE, fill_opacity=0.5)
#             green_layer = Square(side_length=1.5, color=GREEN, fill_opacity=0.5).shift(UP*0.1 + LEFT*0.1)
#             red_layer = Square(side_length=1.5, color=RED, fill_opacity=0.5).shift(UP*0.2 + LEFT*0.2)
            
#             # Group the layers for this batch
#             batch = VGroup(blue_layer, green_layer, red_layer)
            
#             # Position the batch
#             batch_position = LEFT*4.5 + RIGHT*(i * (1.5 + batch_spacing))
#             batch.move_to(batch_position)
            
#             batches.add(batch)

#         # Add dots to indicate more batches
#         if i < 4:  # Only add dots after the third batch
#             dots = Text("...", font_size=40).next_to(batches, RIGHT, buff=0.5)
#             batches.add(dots)
        
#         # Show all batches
#         self.play(FadeIn(batches))
#         self.wait(2)

#         # Create 4 explanation texts to be shown sequentially
#         explanation_text1 = Text("Batch Normalization is applied per channel, across the batch", 
#                               font_size=28).next_to(batches, DOWN, buff=0.7)
#         explanation_text2 = Text("Each channel has its own distribution of values", 
#                               font_size=28).next_to(batches, DOWN, buff=0.7)
#         explanation_text3 = Text("Batch Norm normalizes each channel separately", 
#                               font_size=28).next_to(batches, DOWN, buff=0.7)
#         explanation_text4 = Text("This stabilizes training by reducing internal covariate shift", 
#                               font_size=28).next_to(batches, DOWN, buff=0.7)
        
#         explanation_text4 = Text("One set of μ, σ², γ, β per channel", 
#                               font_size=28).next_to(batches, DOWN, buff=0.7)
        
        
        
#         # Show explanations one after another with transitions
#         self.play(Write(explanation_text1))
#         self.wait(1.5)
#         self.play(FadeOut(explanation_text1), FadeIn(explanation_text2))
#         self.wait(1.5)
#         self.play(FadeOut(explanation_text2), FadeIn(explanation_text3))
#         self.wait(1.5)
#         self.play(FadeOut(explanation_text3), FadeIn(explanation_text4))
#         self.wait(1.5)
        
#         # Remove batches and last text to make room for the next visualization
#         self.play(FadeOut(batches), FadeOut(explanation_text4))

        
#         # THEN: Continue with the original visualization
#         # Create 3 example "channels": R, G, B
#         r_blocks = VGroup(*[Square(color=RED, fill_opacity=1).scale(0.4) for _ in range(4)]).arrange(RIGHT, buff=0.2).shift(2*UP + RIGHT*2)
#         g_blocks = VGroup(*[Square(color=GREEN, fill_opacity=1).scale(0.4) for _ in range(4)]).arrange(RIGHT, buff=0.2).shift(RIGHT*2)
#         b_blocks = VGroup(*[Square(color=BLUE, fill_opacity=1).scale(0.4) for _ in range(4)]).arrange(RIGHT, buff=0.2).shift(2*DOWN + RIGHT*2)

#         self.play(FadeIn(r_blocks), FadeIn(g_blocks), FadeIn(b_blocks))

#         # Slide them to the left
#         self.play(
#             r_blocks.animate.shift(LEFT*4),
#             g_blocks.animate.shift(LEFT*4),
#             b_blocks.animate.shift(LEFT*4),
#             run_time=1.5
#         )

#         # Now add the labels
#         r_label = Text("R Channel", font_size=24).next_to(r_blocks, LEFT)
#         g_label = Text("G Channel", font_size=24).next_to(g_blocks, LEFT)
#         b_label = Text("B Channel", font_size=24).next_to(b_blocks, LEFT)

#         self.play(Write(r_label), Write(g_label), Write(b_label))

#         # Show equations — positioned to the right of the channel blocks
#         equations_group = VGroup(
#             MathTex(r"\text{For each channel compute:}", font_size=30),
#             MathTex(r"\mu_C = \frac{1}{N \cdot H \cdot W} \sum x_{n,c,h,w}", font_size=30),
#             MathTex(r"\sigma^2_C = \frac{1}{N \cdot H \cdot W} \sum (x_{n,c,h,w} - \mu_C)^2", font_size=30),
#             MathTex(r"\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_C}{\sqrt{\sigma^2_C + \epsilon}}", font_size=30),
#             MathTex(r"y_{n,c,h,w} = \gamma_c \, \hat{x}_{n,c,h,w} + \beta_c", font_size=30)
#         ).arrange(DOWN, aligned_edge=LEFT).next_to(g_blocks, RIGHT, buff=1.5)

#         self.play(Write(equations_group[0]))
#         self.wait(0.7)
#         self.play(Write(equations_group[1]))
#         self.wait(0.7)
#         self.play(Write(equations_group[2]))
#         self.wait(0.7)
#         self.play(Write(equations_group[3]))
#         self.wait(0.7)
#         self.play(Write(equations_group[4]))
#         self.wait(2)

#        # Clear screen for numerical example
#         self.play(
#             FadeOut(r_blocks), FadeOut(g_blocks), FadeOut(b_blocks),
#             FadeOut(r_label), FadeOut(g_label), FadeOut(b_label),
#             FadeOut(equations_group),
#             FadeOut(title), FadeOut(batch_text)
#         )
        
#         # Add numerical example title
#         example_title = Text("Batch Normalization: Numerical Example").scale(0.8).to_edge(UP)
#         self.play(Write(example_title))
        
#         # Show numerical example for just the R channel
#         example_text = Text("Red Channel Example (simplified to 1D)", font_size=28).next_to(example_title, DOWN)
#         self.play(Write(example_text))
        
#         # Create a table for original values
#         original_values = [125, 240, 35, 180]  # Example pixel values for red channel
        
#         # Original values table
#         original_table = Table(
#             [["Batch 1", "Batch 2", "Batch 3", "Batch 4"],
#              [str(val) for val in original_values]],
#             row_labels=[Text("Image"), Text("R value")],
#             include_outer_lines=True
#         ).scale(0.6).shift(UP*1)
        
#         self.play(Create(original_table))
        
#         # Step 1: Calculate mean
#         mean_value = sum(original_values) / len(original_values)
#         mean_eq = MathTex(
#             r"\mu_R = \frac{1}{N} \sum x_i = \frac{125 + 240 + 35 + 180}{4} = ", f"{mean_value:.1f}"
#         ).scale(0.8).next_to(original_table, DOWN, buff=0.5)
        
#         self.play(Write(mean_eq))
#         self.wait(1)
        
#         # Step 2: Calculate variance
#         var_values = [(val - mean_value)**2 for val in original_values]
#         variance = sum(var_values) / len(var_values)
        
#         var_eq = MathTex(
#             r"\sigma^2_R = \frac{1}{N} \sum (x_i - \mu_R)^2 = ", f"{variance:.1f}"
#         ).scale(0.8).next_to(mean_eq, DOWN, buff=0.3)
        
#         self.play(Write(var_eq))
#         self.wait(1)
        
#         # Step 3: Normalize (epsilon = 1e-5 for numerical stability)
#         epsilon = 1e-5
#         normalized_values = [(val - mean_value) / (variance + epsilon)**0.5 for val in original_values]
        
#         norm_eq = MathTex(
#             r"\hat{x}_i = \frac{x_i - \mu_R}{\sqrt{\sigma^2_R + \epsilon}}"
#         ).scale(0.8).next_to(var_eq, DOWN, buff=0.3)
        
#         self.play(Write(norm_eq))
#         self.wait(1)
        
#         # Normalized values table
#         normalized_table = Table(
#             [["Batch 1", "Batch 2", "Batch 3", "Batch 4"],
#              [f"{val:.2f}" for val in normalized_values]],
#             row_labels=[Text("Image"), Text("Normalized")],
#             include_outer_lines=True
#         ).scale(0.6).next_to(norm_eq, DOWN, buff=0.5)
        
#         self.play(Create(normalized_table))
#         self.wait(1)
        
#         # Step 4: Scale and shift (γ = 1.5, β = 100)
#         gamma = 1.5
#         beta = 100
#         scaled_values = [gamma * val + beta for val in normalized_values]
        
#         scale_eq = MathTex(
#             r"y_i = \gamma \cdot \hat{x}_i + \beta = ", f"{gamma:.1f}", r" \cdot \hat{x}_i + ", f"{beta}"
#         ).scale(0.8).next_to(normalized_table, DOWN, buff=0.5)
        
#         self.play(Write(scale_eq))
#         self.wait(1)
        
#         # Final values table
#         final_table = Table(
#             [["Batch 1", "Batch 2", "Batch 3", "Batch 4"],
#              [f"{val:.1f}" for val in scaled_values]],
#             row_labels=[Text("Image"), Text("Output")],
#             include_outer_lines=True
#         ).scale(0.6).next_to(scale_eq, DOWN, buff=0.5)
        
#         self.play(Create(final_table))
        
#         # Final explanation
#         final_text = Text("γ and β are learnable parameters that restore representation power", 
#                         font_size=24).next_to(final_table, DOWN, buff=0.5)
#         self.play(Write(final_text))
        
#         self.wait(3)


from manim import *

class BatchNormVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Batch Normalization (Per Channel)").scale(0.7).to_edge(UP)
        self.play(Write(title))

        # Batch info
        batch_text = Text("4 Batches, 3 Channels (R, G, B), 32×32 each", font_size=30).next_to(title, DOWN)
        self.play(FadeIn(batch_text))
        
        # FIRST: Show the batches representation (4 groups of RGB channels)
        batch_spacing = 1.0
        
        # Create 4 batches with RGB channels
        batches = VGroup()
        for i in range(4):
            # Create RGB layers for each batch
            blue_layer = Square(side_length=1.5, color=BLUE, fill_opacity=1)
            green_layer = Square(side_length=1.5, color=GREEN, fill_opacity=1).shift(UP*0.1 + LEFT*0.1)
            red_layer = Square(side_length=1.5, color=RED, fill_opacity=1).shift(UP*0.2 + LEFT*0.2)
            
            # Group the layers for this batch
            batch = VGroup(blue_layer, green_layer, red_layer)
            
            # Position the batch
            batch_position = LEFT*4.5 + RIGHT*(i * (1.5 + batch_spacing))
            batch.move_to(batch_position)
            
            batches.add(batch)
        
        # Add dots to indicate more batches
        if i < 3:  # Only add dots after the third batch
            dots = Text("...", font_size=40).next_to(batches, RIGHT, buff=0.5)
            batches.add(dots)
        
        # Show all batches
        self.play(FadeIn(batches))
        
        # Create 4 explanation texts to be shown sequentially
        explanation_text1 = Text("Input data consists of multiple images with RGB channels", 
                              font_size=28).next_to(batches, DOWN, buff=0.7)
        explanation_text2 = Text("Each channel has its own distribution of values", 
                              font_size=28).next_to(batches, DOWN, buff=0.7)
        explanation_text3 = Text("Batch Norm normalizes each channel separately", 
                              font_size=28).next_to(batches, DOWN, buff=0.7)
        explanation_text4 = Text("This stabilizes training by reducing internal covariate shift", 
                              font_size=28).next_to(batches, DOWN, buff=0.7)
        
        # Show explanations one after another with transitions
        self.play(Write(explanation_text1))
        self.wait(1.5)
        self.play(FadeOut(explanation_text1), FadeIn(explanation_text2))
        self.wait(1.5)
        self.play(FadeOut(explanation_text2), FadeIn(explanation_text3))
        self.wait(1.5)
        self.play(FadeOut(explanation_text3), FadeIn(explanation_text4))
        self.wait(1.5)
        
        # Remove batches and last text to make room for the next visualization
        self.play(FadeOut(batches), FadeOut(explanation_text4))
        
        # THEN: Continue with the original visualization
        # Create 3 example "channels": R, G, B
        r_blocks = VGroup(*[Square(color=RED, fill_opacity=1).scale(0.4) for _ in range(4)]).arrange(RIGHT, buff=0.2).shift(2*UP + RIGHT*2)
        g_blocks = VGroup(*[Square(color=GREEN, fill_opacity=1).scale(0.4) for _ in range(4)]).arrange(RIGHT, buff=0.2).shift(RIGHT*2)
        b_blocks = VGroup(*[Square(color=BLUE, fill_opacity=1).scale(0.4) for _ in range(4)]).arrange(RIGHT, buff=0.2).shift(2*DOWN + RIGHT*2)

        self.play(FadeIn(r_blocks), FadeIn(g_blocks), FadeIn(b_blocks))

        # Slide them to the left
        self.play(
            r_blocks.animate.shift(LEFT*4),
            g_blocks.animate.shift(LEFT*4),
            b_blocks.animate.shift(LEFT*4),
            run_time=1.5
        )

        # Now add the labels
        r_label = Text("R Channel", font_size=24).next_to(r_blocks, LEFT)
        g_label = Text("G Channel", font_size=24).next_to(g_blocks, LEFT)
        b_label = Text("B Channel", font_size=24).next_to(b_blocks, LEFT)

        self.play(Write(r_label), Write(g_label), Write(b_label))

        # Show equations — positioned to the right of the channel blocks
        equations_group = VGroup(
            MathTex(r"\text{For each channel compute:}", font_size=30),
            MathTex(r"\mu_C = \frac{1}{N \cdot H \cdot W} \sum x_{n,c,h,w}", font_size=30),
            MathTex(r"\sigma^2_C = \frac{1}{N \cdot H \cdot W} \sum (x_{n,c,h,w} - \mu_C)^2", font_size=30),
            MathTex(r"\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_C}{\sqrt{\sigma^2_C + \epsilon}}", font_size=30),
            MathTex(r"y_{n,c,h,w} = \gamma_c \, \hat{x}_{n,c,h,w} + \beta_c", font_size=30)
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
            FadeOut(equations_group),
            FadeOut(title), FadeOut(batch_text)
        )
        
        # Add numerical example title
        example_title = Text("Batch Normalization: Numerical Example").scale(0.8).to_edge(UP)
        self.play(Write(example_title))
        
        # Show numerical example for just the R channel
        example_text = Text("Red Channel Example (simplified to 1D)", font_size=28).next_to(example_title, DOWN)
        self.play(Write(example_text))
        
        # Create a table for original values
        original_values = [125, 240, 35, 180]  # Example pixel values for red channel
        
        # Original values table
        original_table = Table(
            [["Batch 1", "Batch 2", "Batch 3", "Batch 4"],
             [str(val) for val in original_values]],
            row_labels=[Text("Image"), Text("R value")],
            include_outer_lines=True
        ).scale(0.6).shift(UP*1)
        
        self.play(Create(original_table))
        
        # Step 1: Calculate mean
        mean_value = sum(original_values) / len(original_values)
        mean_eq = MathTex(
            r"\mu_R = \frac{1}{N} \sum x_i = \frac{125 + 240 + 35 + 180}{4} = ", f"{mean_value:.1f}"
        ).scale(0.8).next_to(original_table, DOWN, buff=0.5)
        
        self.play(Write(mean_eq))
        self.wait(1)
        
        # Step 2: Calculate variance
        var_values = [(val - mean_value)**2 for val in original_values]
        variance = sum(var_values) / len(var_values)
        
        var_eq = MathTex(
            r"\sigma^2_R = \frac{1}{N} \sum (x_i - \mu_R)^2 = ", f"{variance:.1f}"
        ).scale(0.8).next_to(mean_eq, DOWN, buff=0.3)
        
        self.play(Write(var_eq))
        self.wait(1)
        
        
        
        # Step 3: Normalize (epsilon = 1e-5 for numerical stability)
        epsilon = 1e-5
        normalized_values = [(val - mean_value) / (variance + epsilon)**0.5 for val in original_values]
        
        norm_eq = MathTex(
            r"\hat{x}_i = \frac{x_i - \mu_R}{\sqrt{\sigma^2_R + \epsilon}}"
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
        norm_screen_title = Text("Normalized R Channel Values").scale(0.8).to_edge(UP)
        self.play(Write(norm_screen_title))
        
        
        
        # Show the normalized values in a table
        norm_table = Table(
            [["Batch 1", "Batch 2", "Batch 3", "Batch 4"],
             [f"{val:.2f}" for val in normalized_values]],
            row_labels=[Text("Image"), Text("Normalized")],
            include_outer_lines=True
        ).scale(0.6).next_to(norm_screen_title, DOWN, buff=0.5)
        
        self.play(Create(norm_table))
        self.wait(1)
        
        # Step 4: Scale and shift (γ = 1.5, β = 100)
        gamma = 1.5
        beta = 100
        scaled_values = [gamma * val + beta for val in normalized_values]
        
        scale_eq = MathTex(
            r"y_i = \gamma \cdot \hat{x}_i + \beta = ", f"{gamma:.1f}", r" \cdot \hat{x}_i + ", f"{beta}"
        ).scale(0.8).next_to(norm_table, DOWN, buff=0.5)
        
        self.play(Write(scale_eq))
        self.wait(1)
        
        # Final values table
        final_table = Table(
            [["Batch 1", "Batch 2", "Batch 3", "Batch 4"],
             [f"{val:.1f}" for val in scaled_values]],
            row_labels=[Text("Image"), Text("Output")],
            include_outer_lines=True
        ).scale(0.6).next_to(scale_eq, DOWN, buff=0.5)
        
        self.play(Create(final_table))
        
        # Final explanation
        final_text = Text("γ and β are learnable parameters that restore representation power", 
                        font_size=24).next_to(final_table, DOWN, buff=0.5)
        self.play(Write(final_text))
        
        
        
        
        
        # Clear previous screen (final_table and final_text)
        self.play(
            FadeOut(norm_screen_title),
            FadeOut(norm_table),
            FadeOut(final_table),
            FadeOut(scale_eq),
            FadeOut(final_text)
        )

        # New screen: What Happens During Training vs Testing?
        headline = Text("🧪 What Happens During Training vs Testing in Batch Normalization?", font_size=30).to_edge(UP)
        self.play(Write(headline))

        # Training Section
        training_title = Text("🟢 During Training:", font_size=28,color=RED).next_to(headline, DOWN, aligned_edge=LEFT).shift(DOWN*0.5)
        training_points = VGroup(
            Text("• Computes mean and variance from current mini-batch.", font_size=24),
            Text("• Normalizes using these batch stats.", font_size=24),
            Text("• Maintains a running average of mean and variance.", font_size=24),
            Text('• Like: “Use this batch’s average now, but remember it for later.”', font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(training_title, DOWN, aligned_edge=LEFT)

        self.play(Write(training_title), FadeIn(training_points, shift=RIGHT))
        self.wait(2)

        # Testing Section
        testing_title = Text("🔵 During Testing (Inference):", font_size=28,color=RED).next_to(training_points, DOWN, aligned_edge=LEFT).shift(DOWN*0.7)
        testing_points = VGroup(
            Text("• No mini-batch stats used.", font_size=24),
            Text("• Uses the running averages saved during training.", font_size=24),
            Text("• Ensures stable, consistent outputs.", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(testing_title, DOWN, aligned_edge=LEFT)

        self.play(Write(testing_title), FadeIn(testing_points, shift=RIGHT))
        
        # Clear previous screen (final_table and final_text)
        self.play(
            FadeOut(training_title),
            FadeOut(training_points),
            FadeOut(testing_title),
            FadeOut(testing_points),
            FadeOut(headline)
            )
        
    
        # Title for second table
        bn_feature_title = Text("📋 Batch Normalization – Key Characteristics").scale(0.7).to_edge(UP)
        self.play(Write(bn_feature_title))

        # Define table data
        bn_features = [
            ["Feature", "Batch Normalization"],
            ["Where it normalizes", "Across the batch dimension (mini-batch)"],
            ["Formula", "Normalizes per feature across samples"],
            ["Best for", "CNNs, deep feedforward nets"],
            ["Placement", "After Conv or Linear, before activation"],
            ["Mini-batch dependency", "✅ Yes – needs large batch size for stability"],
            ["During inference", "Uses running averages of mean and variance"],
            ["Learnable parameters", "✅ Yes (γ, β)"],
            ["GPU efficiency", "✅ Highly optimized"],
        ]

        # Create table
        bn_table = Table(
            [row for row in bn_features[1:]],  # exclude header for content
            col_labels=[Text(bn_features[0][0], font_size=26), Text(bn_features[0][1], font_size=26)],
            element_to_mobject=lambda s: Text(s, font_size=18, line_spacing=0.8),
            include_outer_lines=True
        ).scale(0.7).next_to(bn_feature_title, DOWN, buff=0.4)

        self.play(Create(bn_table))
        self.wait(4)
                
        
        self.wait(3)
        