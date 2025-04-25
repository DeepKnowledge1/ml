from manim import *
import numpy as np  # Wichtig: NumPy importieren, da es verwendet wird
# from manim import Scene, Text, VGroup, Circle, Line, Axes, Dot, DashedLine, MathTex, Create, Write, FadeOut, FadeIn, AnimationGroup, MoveAlongPath, ORIGIN, UP, DOWN, LEFT, RIGHT, BLUE, RED, GREEN, YELLOW, GRAY, PURPLE, linear

class DropoutVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Understanding Dropout in Neural Networks", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.scale(0.6).to_edge(UP))
        
        # Introduction to Neural Network
        self.explain_neural_network_basics()
        
        # Overfitting explanation
        self.explain_overfitting()
        
        # Dropout mechanism demonstration
        self.demonstrate_dropout()
        
        # Training vs Inference
        self.show_training_vs_inference()
        
        # Benefits and conclusion
        self.show_benefits()
        
        self.wait(2)
    
    def explain_neural_network_basics(self):
        # Create a simple neural network
        network = self.create_network()
        network_label = Text("Standard Neural Network", font_size=24).next_to(network, DOWN)
        
        self.play(Create(network), Write(network_label))
        self.wait(2)
        
        # Explain forward pass
        self.explain_forward_pass(network)
        
        self.play(FadeOut(network), FadeOut(network_label))
    
    def create_network(self, layer_neurons=[4, 6, 6, 4], spacing=0.7):
        layers = []
        
        # Create all neurons as circles
        for i, n_neurons in enumerate(layer_neurons):
            layer = VGroup()
            for j in range(n_neurons):
                neuron = Circle(radius=0.2, color=BLUE, fill_opacity=0.5)
                neuron.move_to([i*spacing*3, (j - (n_neurons-1)/2)*spacing, 0])
                layer.add(neuron)
            layers.append(layer)
        
        # Create connections between layers
        connections = VGroup()
        for l1, l2 in zip(layers[:-1], layers[1:]):
            for n1 in l1:
                for n2 in l2:
                    connection = Line(n1.get_center(), n2.get_center(), 
                                    stroke_width=1, stroke_opacity=0.5)
                    connections.add(connection)
        
        # Group everything together
        network = VGroup(*layers, connections)
        network.scale(1.4).move_to(ORIGIN)
        
        return network
    
    def explain_forward_pass(self, network):
        # Highlight the data flow through the network
        pulses = []
        for layer_idx in range(3):  # For 3 layer transitions
            layer_pulses = []
            for connection in network[4]:  # Index 4 is the connections group
                if abs(connection.start[0] - layer_idx) < 0.1:
                    pulse = Dot(color=YELLOW)
                    pulse.move_to(connection.start)
                    anim = MoveAlongPath(pulse, connection, run_time=1.5)
                    layer_pulses.append(AnimationGroup(anim, rate_func=linear))
            
            # Stagger the animations within each layer
            if layer_pulses:
                pulses.append(AnimationGroup(*layer_pulses))
        
        # Play the pulse animations in sequence
        for pulse_group in pulses:
            self.play(pulse_group)
            self.wait(0.5)
    
    def explain_overfitting(self):
        # Create a simple graph showing overfitting
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": BLUE},
        ).scale(0.6)
        
        # Create data points
        points = VGroup()
        point_coords = [(1, 1), (2, 3), (3, 2), (4, 4), (5, 3), 
                        (6, 5), (7, 7), (8, 6), (9, 8)]
        
        for x, y in point_coords:
            point = Dot(axes.coords_to_point(x, y), color=RED)
            points.add(point)
        
        # Create simple line (good fit)
        line = axes.plot(lambda x: 0.8*x + 0.2, color=GREEN)
        
        # Create complex curve (overfit)
        overfit = axes.plot(lambda x: 0.8*x + 0.5*np.sin(1.5*x) + 0.2, color=YELLOW)
        
        # Labels
        graph_title = Text("Overfitting Example", font_size=28).next_to(axes, UP)
        train_label = Text("Training Data", font_size=20, color=RED).to_edge(LEFT).shift(UP*1)
        good_fit = Text("Good Model", font_size=20, color=GREEN).to_edge(LEFT)
        overfit_label = Text("Overfit Model", font_size=20, color=YELLOW).to_edge(LEFT).shift(DOWN*1)
        
        # Group everything
        graph_group = VGroup(axes, points, graph_title)
        labels = VGroup(train_label, good_fit, overfit_label)
        
        # Position the graph
        graph_group.move_to(ORIGIN)
        
        # Show the graph with points
        self.play(Create(axes), Write(graph_title))
        self.play(Create(points))
        self.add(train_label)
        self.wait(1)
        
        # Add the good fit line
        self.play(Create(line))
        self.add(good_fit)
        self.wait(1.5)
        
        # Add the overfit curve
        self.play(Create(overfit))
        self.add(overfit_label)
        self.wait(2)
        
        # Add new test points
        test_points = VGroup()
        test_coords = [(2.5, 1.8), (3.5, 3.3), (5.5, 4.2), (7.5, 6.5), (8.5, 7.2)]
        
        for x, y in test_coords:
            point = Dot(axes.coords_to_point(x, y), color=BLUE)
            test_points.add(point)
        
        test_label = Text("Test Data", font_size=20, color=BLUE).to_edge(RIGHT).shift(UP*1)
        self.play(Create(test_points))
        self.add(test_label)
        
        # Show prediction errors
        for point in test_points:
            x, y, = axes.point_to_coords(point.get_center())

            # Error line for overfit model
            overfit_y = 0.8*x + 0.5*np.sin(1.5*x) + 0.2
            overfit_point = axes.coords_to_point(x, overfit_y)
            overfit_line = DashedLine(point.get_center(), overfit_point, color=YELLOW)
            
            # Error line for good model
            good_y = 0.8*x + 0.2
            good_point = axes.coords_to_point(x, good_y)
            good_line = DashedLine(point.get_center(), good_point, color=GREEN)
            
            self.play(Create(good_line), Create(overfit_line), run_time=0.5)
        
        self.wait(2)
        
        conclusion = Text("Overfitting: Perfect on training data, poor on new data", 
                           font_size=24, color=YELLOW).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)
        
        # Clean up
        self.play(
            FadeOut(graph_group),
            FadeOut(line),
            FadeOut(overfit),
            FadeOut(labels),
            FadeOut(test_points),
            FadeOut(test_label),
            FadeOut(conclusion),
            FadeOut(good_line),
            FadeOut(overfit_line)
        )
    
    def demonstrate_dropout(self):
        # Create a standard network and a network with dropout
        standard_network = self.create_network()
        standard_network.move_to(LEFT*3)
        
        dropout_network = self.create_network()
        dropout_network.move_to(RIGHT*3)
        
        standard_label = Text("Standard Network", font_size=24).next_to(standard_network, DOWN)
        dropout_label = Text("With Dropout (p=0.5)", font_size=24).next_to(dropout_network, DOWN)
        
        self.play(
            Create(standard_network),
            Create(dropout_network),
            Write(standard_label),
            Write(dropout_label)
        )
        self.wait(1)
        
        # Demonstrate dropout by fading out random neurons
        dropout_prob = 0.5
        for i in range(1, 3):  # Apply to hidden layers only
            layer = dropout_network[i]
            for neuron in layer:
                if np.random.random() < dropout_prob:
                    self.play(neuron.animate.set_fill(GRAY, opacity=0.1), run_time=0.3)
        
        self.wait(2)
        
        # Highlight the remaining active paths
        active_paths = VGroup()
        for connection in dropout_network[4]:  # connections
            start_neuron = None
            end_neuron = None
            
            # Find the neurons this connection connects
            for layer in dropout_network[:4]:  # all neuron layers
                for neuron in layer:
                    if np.isclose(connection.start, neuron.get_center(), atol=0.1).all():
                        start_neuron = neuron
                    if np.isclose(connection.end, neuron.get_center(), atol=0.1).all():
                        end_neuron = neuron
            
            # If both neurons are active, highlight the connection
            if start_neuron and end_neuron and \
               start_neuron.get_fill_opacity() > 0.3 and end_neuron.get_fill_opacity() > 0.3:
                active_paths.add(connection)
        
        self.play(active_paths.animate.set_stroke(YELLOW, opacity=1, width=2))
        self.wait(2)
        
        # Add explanation text
        dropout_text = Text(
            "During each training batch, neurons are randomly disabled",
            font_size=24
        ).to_edge(UP, buff=1.5)
        
        self.play(Write(dropout_text))
        self.wait(2)
        
        # Show a new batch with different dropout pattern
        reset_network = self.create_network()
        reset_network.move_to(RIGHT*3)
        
        self.play(
            FadeOut(dropout_network),
            FadeIn(reset_network)
        )
        
        dropout_network = reset_network
        
        # Apply new dropout pattern
        for i in range(1, 3):  # Apply to hidden layers only
            layer = dropout_network[i]
            for neuron in layer:
                if np.random.random() < dropout_prob:
                    self.play(neuron.animate.set_fill(GRAY, opacity=0.1), run_time=0.2)
        
        dropout_text2 = Text(
            "Different neurons are dropped in each training batch",
            font_size=24
        ).next_to(dropout_text, DOWN)
        
        self.play(Write(dropout_text2))
        self.wait(3)
        
        # Clean up
        self.play(
            FadeOut(standard_network),
            FadeOut(dropout_network),
            FadeOut(standard_label),
            FadeOut(dropout_label),
            FadeOut(dropout_text),
            FadeOut(dropout_text2)
        )
    
    def show_training_vs_inference(self):
        # Create title
        section_title = Text("Dropout: Training vs. Inference", font_size=32)
        self.play(Write(section_title))
        self.wait(1)
        self.play(section_title.animate.scale(0.8).to_edge(UP))
        
        # Create networks for training and inference
        training_network = self.create_network(layer_neurons=[3, 4, 4, 2], spacing=0.8)
        training_network.scale(0.9).move_to(LEFT*3.5)
        
        inference_network = self.create_network(layer_neurons=[3, 4, 4, 2], spacing=0.8)
        inference_network.scale(0.9).move_to(RIGHT*3.5)
        
        training_label = Text("Training Phase", font_size=24).next_to(training_network, DOWN)
        inference_label = Text("Inference Phase", font_size=24).next_to(inference_network, DOWN)
        
        self.play(
            Create(training_network),
            Create(inference_network),
            Write(training_label),
            Write(inference_label)
        )
        self.wait(1)
        
        # Apply dropout to training network
        dropout_prob = 0.5
        dropped_neurons = []
        for i in range(1, 3):  # Apply to hidden layers only
            layer = training_network[i]
            for neuron in layer:
                if np.random.random() < dropout_prob:
                    self.play(neuron.animate.set_fill(GRAY, opacity=0.1), run_time=0.2)
                    dropped_neurons.append(neuron)
        
        # Add scaling indicators to inference network
        scaling_factors = []
        for i in range(1, 3):  # Apply to hidden layers only
            layer = inference_network[i]
            for neuron in layer:
                # scale_text = MathTex(r"\times 0.5", font_size=16, color=YELLOW)
                scale_text = Text("× 0.5", font_size=16, color=YELLOW)

                scale_text.next_to(neuron, UP, buff=0.1)
                scaling_factors.append(scale_text)
                self.play(Write(scale_text), run_time=0.2)
        
        # Add explanation
        training_text = Text("Random neurons disabled", font_size=20).next_to(training_network, UP)
        inference_text = Text("All neurons active but outputs scaled", font_size=20).next_to(inference_network, UP)
        
        self.play(
            Write(training_text),
            Write(inference_text)
        )
        self.wait(2)
        
        # Show formula for inverted dropout
        # formula = MathTex(
        #     r"\text{mask} \sim \text{Bernoulli}(p)",
        #     r"\\",
        #     r"y_{\text{dropout}} = y \cdot \text{mask} / (1-\text{dropout\_rate})"
        # ).scale(0.8).to_edge(DOWN)
        
        formula = Text(
            "mask ~ Bernoulli(p)\ny_dropout = y × mask / (1 - dropout_rate)",
            font_size=20
        ).to_edge(DOWN)
        
        self.play(Write(formula))
        self.wait(3)
        
        # Clean up
        self.play(
            FadeOut(training_network),
            FadeOut(inference_network),
            FadeOut(training_label),
            FadeOut(inference_label),
            FadeOut(training_text),
            FadeOut(inference_text),
            FadeOut(formula),
            FadeOut(section_title),
            *[FadeOut(sf) for sf in scaling_factors]
        )
    

    def show_benefits(self):
        # Create title
        benefits_title = Text("Benefits of Dropout", font_size=32)
        self.play(Write(benefits_title))
        self.wait(1)
        self.play(benefits_title.animate.scale(0.8).to_edge(UP))
        
        # Create a network with highlighted features
        network = self.create_network(layer_neurons=[3, 4, 4, 2], spacing=0.8)
        network.scale(1.2).move_to(LEFT*3)
        
        self.play(Create(network))
        self.wait(1)
        
        # Highlight different paths through the network
        paths = [
            [network[0][0], network[1][0], network[2][0], network[3][0]],  # Path 1
            [network[0][1], network[1][1], network[2][1], network[3][0]],  # Path 2
            [network[0][2], network[1][3], network[2][3], network[3][1]],  # Path 3
        ]
        
        path_colors = [YELLOW, GREEN, PURPLE]
        
        for i, path in enumerate(paths):
            path_group = VGroup(*path)
            self.play(path_group.animate.set_fill(path_colors[i], opacity=0.8), run_time=1)
            
            # Add connections between the path neurons
            for j in range(len(path)-1):
                connection = Line(
                    path[j].get_center(),
                    path[j+1].get_center(),
                    stroke_width=3,
                    stroke_color=path_colors[i]
                )
                self.play(Create(connection), run_time=0.5)
            
            self.wait(0.5)
        
        # Add benefits text
        benefits = [
            "1. Prevents co-adaptation of neurons",
            "2. Forces redundant feature learning",
            "3. Acts like an ensemble of networks",
            "4. Improves generalization"
        ]
        
        benefits_group = VGroup()
        for i, text in enumerate(benefits):
            benefit = Text(text, font_size=24)
            benefit.move_to(RIGHT*3 + DOWN*(i*0.8 - 1.2))
            benefits_group.add(benefit)
            self.play