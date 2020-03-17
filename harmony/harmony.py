import os
import tensorflow as tf
import numpy
import sys

from initializers import Gravitate, Resonate, LinearGradient
from util import LinearAttractionFromLog

script_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(script_dir, "models")
checkpoints_dir = os.path.join(script_dir, "checkpoints")


class AlgHarmony:
    def __init__(self):
        self.session = tf.compat.v1.Session()
        self.tensorboard = True

        num_nodes = self.num_nodes = 120

        self.setup_variables()

        self.build()

        self.init = tf.compat.v1.variables_initializer(
            tf.compat.v1.global_variables(), name="init_all_vars_op"
        )

        self.export_test()

    def setup_variables(self):

        self.potentials = tf.Variable(numpy.zeros(self.num_nodes), dtype=tf.float32)
        self.inhibitions = tf.Variable(numpy.zeros(self.num_nodes), dtype=tf.float32)
        self.players = tf.Variable(numpy.ones(self.num_nodes), dtype=tf.float32)
        self.energies = tf.Variable(numpy.zeros(self.num_nodes), dtype=tf.float32)

    def export_test(self):

        tf.compat.v1.train.write_graph(
            self.session.graph_def, "/opt/alg-harmony", "graph.pb"
        )

    def build(self):

        inputs = self.build_input()

        replenished_players = self.replenish_players(self.players)

        # Resonate the input to potential
        self.build_resonate_layer()
        resonated_inputs = self.resonate(
            inputs, self.resonate_weights, self.resonate_gain
        )

        tf.summary.histogram("inputs", inputs)
        resonated_inputs = resonated_inputs + inputs
        seeded_potentials = self.seed_potential(self.potentials, resonated_inputs)

        # Move and space out players
        moved_players = self.move_players(replenished_players, seeded_potentials)
        spaced_players = self.space_players(moved_players)

        # Players react with (eat) potential to produce energy
        remaining_potential, energy_produced = self.react(
            spaced_players, seeded_potentials
        )
        with tf.name_scope("update_energies"):
            new_energies = self.energies + ((1 - self.energies) * energy_produced)

        # Resonate new energy to potential
        resonated_potentials = self.resonate(
            new_energies, self.resonate_weights, self.resonate_gain
        )
        reverse_resonated_potentials = self.reverse_resonate(
            new_energies, self.resonate_weights, self.reverse_resonate_gain
        )
        fully_resonated_potentials = (
            resonated_potentials + reverse_resonated_potentials + energy_produced
        )

        # Update and apply inhibitions
        updated_inhibitions = self.update_inhibitions(new_energies, self.inhibitions)
        inhibited_potential = self.inhibit_potential(
            fully_resonated_potentials, updated_inhibitions
        )

        # Attract new potential to already existing potential
        attracted_potentials = self.attract_potential(
            remaining_potential, inhibited_potential, 1
        )

        # Adjust the brightness (how many high notes there are)
        self.build_brightness_layer()
        brighten_energies = self.adjust_brightness(new_energies)
        brighten_potential = self.adjust_brightness(attracted_potentials)

        # Decay the variables
        self.build_decay_layer()
        decayed_potentials = self.decay(brighten_potential, self.potential_decay)
        decayed_energies = self.decay(brighten_energies, self.energy_decay)
        decayed_players = self.decay(spaced_players, self.play_decay)

        # Update internal variables
        self.potentials = tf.compat.v1.assign(self.potentials, decayed_potentials)
        self.inhibitions = tf.compat.v1.assign(self.inhibitions, updated_inhibitions)
        self.players = tf.compat.v1.assign(self.players, decayed_players)
        self.energies = tf.compat.v1.assign(self.energies, decayed_energies)

        mean_poten = tf.reduce_mean(self.potentials)
        mean_inhib = tf.reduce_mean(self.inhibitions)
        mean_play = tf.reduce_mean(self.players)
        mean_energ = tf.reduce_mean(self.energies)
        self.potentials = tf.compat.v1.Print(
            self.potentials,
            [mean_energ, mean_inhib, mean_play, mean_poten],
            message="E - I - Pl - Po: ",
            summarize=self.num_nodes,
        )

        tf.summary.histogram("hist_potentials", self.potentials)
        tf.summary.histogram("hist_inhibitions", self.inhibitions)
        tf.summary.histogram("hist_players", self.players)
        tf.summary.histogram("hist_energies", self.energies)

        # Setup outputs
        tf.identity(self.potentials, name="potentials")
        tf.identity(self.inhibitions, name="inhibitions")
        tf.identity(self.players, name="players")
        tf.identity(self.energies, name="energies")

    # All the bits and bobs broken down into operations

    def build_input(self, name="inputs"):
        with tf.name_scope(name if self.tensorboard else None):
            self.audio_inputs = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(self.num_nodes), name="audio_inputs"
            )
            self.midi_inputs = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(self.num_nodes), name="midi_inputs"
            )

            self.audio_input_gain = tf.Variable(
                0.8, name="audio_input_gain", dtype=tf.float32
            )
            self.midi_input_gain = tf.Variable(
                0.8, name="midi_input_gain", dtype=tf.float32
            )

            return (self.midi_inputs * self.midi_input_gain) + (
                self.audio_inputs * self.audio_input_gain
            )

    def replenish_players(self, play, name="replenish_play"):
        with tf.name_scope(name if self.tensorboard else None):
            self.play_replenishment = tf.Variable(
                0.8, name="play_replenishment", dtype=tf.float32
            )
            return play + (self.play_replenishment * (1 - play))

    def build_resonate_layer(self, name="res_layer"):
        with tf.name_scope(name if self.tensorboard else None):
            self.resonate_weights = tf.Variable(
                Resonate()([self.num_nodes, self.num_nodes]),
                # expected_shape=[self.num_nodes, self.num_nodes],
                name="resonate_weights",
                dtype=tf.float32,
            )
            self.resonate_gain = tf.Variable(
                0.7, name="resonate_gain", dtype=tf.float32
            )
            self.reverse_resonate_gain = tf.Variable(
                0.7, name="reverse_resonate_gain", dtype=tf.float32
            )

    def seed_potential(self, potential, inputs, name="seed_potential"):
        with tf.name_scope(name if self.tensorboard else None):
            return potential + ((1 - potential) * inputs)

    def update_inhibitions(self, energies, inhibitions, name="update_inhibs"):
        # Move each node's inhibition level, towards the level of the energy on that node,
        # proportional to inhibition_rate
        with tf.name_scope(name if self.tensorboard else None):
            self.inhibition_rate = tf.Variable(
                0.01, name="inhibition_rate", dtype=tf.float32
            )
            return inhibitions + (
                tf.pow(energies - inhibitions, 2) * (self.inhibition_rate * 0.000001)
            )

    def inhibit_potential(self, potentials, inhibitions, name="inhib_potential"):
        with tf.name_scope(name if self.tensorboard else None):
            self.inhibition_gain = tf.Variable(
                0.1, name="inhibition_gain", dtype=tf.float32
            )
            attraction_gravity = Gravitate(
                get_attraction=LinearAttractionFromLog(range_in_hz=80)
            )
            self.inhibition_averaging_weights = tf.Variable(
                attraction_gravity([self.num_nodes, self.num_nodes]),
                name="inhibition_averaging_weights",
            )

            potential_local_averages = tf.einsum(
                "n,nm->m", potentials, self.inhibition_averaging_weights
            )

            # Remove potential from a node proportional to its inhibition level and inhibition_gain
            return potentials * (
                1 - (inhibitions * self.inhibition_gain * potential_local_averages)
            )

    def move_players(self, players, potentials, name="move_players"):
        with tf.name_scope(name if self.tensorboard else None):
            search_gravity = Gravitate()
            self.player_search_weights = tf.Variable(
                search_gravity([self.num_nodes, self.num_nodes]),
                name="player_search_weights",
            )
            self.play_speed = tf.Variable(0.1, name="play_speed", dtype=tf.float32)

            # Move players towards potential
            moving_players, remaining_players = self.gravitate(
                potentials, players, self.player_search_weights, self.play_speed
            )
            return tf.clip_by_value(
                remaining_players + ((1 - remaining_players) * moving_players), 0, 1
            )

    def space_players(self, players, name="space_players"):
        with tf.name_scope(name if self.tensorboard else None):
            self.play_spacing = tf.Variable(0.1, name="play_spacing", dtype=tf.float32)

            linear_attraction_from_log_exclude = LinearAttractionFromLog(
                range_in_hz=30, exclude_same_index=True
            )
            attraction_gravity = Gravitate(
                get_attraction=linear_attraction_from_log_exclude
            )
            self.player_spacing_weights = tf.Variable(
                attraction_gravity([self.num_nodes, self.num_nodes]),
                name="player_spacing_weights",
            )

            return self.space(
                self.energies, players, self.player_spacing_weights, self.play_spacing
            )

    def react(self, players, potentials, name="react"):
        with tf.name_scope(name if self.tensorboard else None):
            self.play_appetite = tf.Variable(
                0.1, name="play_appetite", dtype=tf.float32
            )
            self.play_vigor = tf.Variable(0.7, name="play_vigor", dtype=tf.float32)

            consumption = tf.pow(players, 2) * self.play_appetite
            consumed_potentials = tf.pow(potentials, 2) * consumption
            potentials = potentials - consumed_potentials
            energy = consumed_potentials * self.play_vigor
            return potentials, energy

    def reverse_resonate(self, input, weights, gain, name="reverse_resonate"):
        with tf.name_scope(name if self.tensorboard else None):
            transposed_weights = tf.transpose(weights)
            inverse_weights = tf.where(
                tf.less(transposed_weights, 0.05),
                0.0 * transposed_weights,
                1.0 / transposed_weights,
            )
            resonated = tf.einsum("n,nm->m", input, inverse_weights) * gain * 0.01
            return resonated

    def attract_potential(
        self, potential, new_potential, strength, name="attract_potentials"
    ):
        with tf.name_scope(name if self.tensorboard else None):
            attraction_gravity = Gravitate(
                get_attraction=LinearAttractionFromLog(range_in_hz=15)
            )
            self.potential_grouping_weights = tf.Variable(
                attraction_gravity([self.num_nodes, self.num_nodes]),
                name="potential_grouping_weights",
            )
            self.potential_grouping = tf.Variable(
                0.7, name="potential_grouping", dtype=tf.float32
            )

            moving_potentials, remaining_potentials = self.gravitate(
                potential, new_potential, self.potential_grouping_weights, strength
            )

            return tf.clip_by_value(
                potential
                + (
                    (1 - potential)
                    * (
                        (1 - self.potential_grouping) * remaining_potentials
                        + (self.potential_grouping * moving_potentials)
                    )
                ),
                0,
                1,
            )

    def build_brightness_layer(self, name="brightness_layer"):
        with tf.name_scope(name if self.tensorboard else None):
            linear_gradient = LinearGradient()
            self.brightnesses = tf.Variable(
                linear_gradient([self.num_nodes]), name="brightnesses"
            )
            self.brightness = tf.Variable(0.95, name="brightness", dtype=tf.float32)

    def adjust_brightness(self, inputs, name="adjust_brightness"):
        with tf.name_scope(name if self.tensorboard else None):
            filtered = inputs * self.brightnesses
            return (self.brightness * inputs) + ((1 - self.brightness) * filtered)

    def build_decay_layer(self, name="decay_layer"):
        with tf.name_scope(name if self.tensorboard else None):
            attraction_gravity = Gravitate(
                get_attraction=LinearAttractionFromLog(range_in_hz=80)
            )
            self.decay_averaging_weights = tf.Variable(
                attraction_gravity([self.num_nodes, self.num_nodes]),
                name="decay_averaging_weights",
            )

            self.potential_decay = tf.Variable(
                0.8, name="potential_decay", dtype=tf.float32
            )
            self.play_decay = tf.Variable(0.8, name="play_decay", dtype=tf.float32)
            self.energy_decay = tf.Variable(0.95, name="energy_decay", dtype=tf.float32)

    def decay(self, values, strength, name="decay"):
        with tf.name_scope(name if self.tensorboard else None):
            local_averages = tf.einsum("n,nm->m", values, self.decay_averaging_weights)
            return tf.clip_by_value(values - (values * strength * local_averages), 0, 1)

    def export_metrics(self, run=0):
        LOGDIR = "/opt/alg-harmony/logs/summaries/" + str(run)
        writer = tf.summary.FileWriter(LOGDIR)
        writer.add_graph(self.session.graph)

        self.session.run(tf.global_variables_initializer())

        fetches = [
            self.audio_inputs,
            self.midi_inputs,
            self.energies,
            self.potentials,
            self.players,
            self.inhibitions,
            self.inhibition_rate,
            self.inhibition_gain,
            self.inhibition_averaging_weights,
        ]

        for i in range(20000):
            self.session.run(
                fetches,
                feed_dict={
                    self.audio_inputs: numpy.random.random(self.num_nodes),
                    self.midi_inputs: numpy.zeros(self.num_nodes),
                },
            )

            if i % 50 == 0:
                summary = self.session.run(
                    self.merged_summary,
                    feed_dict={
                        self.audio_inputs: numpy.random.random(self.num_nodes),
                        self.midi_inputs: numpy.zeros(self.num_nodes),
                    },
                )
                writer.add_summary(summary, i)
        writer.flush()
        writer.close()

    def export(self):
        tf.compat.v1.train.write_graph(
            self.session.graph_def, "/opt/alg-harmony", "graph.pb"
        )
        # summary_writer = tf.summary.FileWriter("./tensorboard-example/", graph=tf.get_default_graph())
        LOGDIR = "/opt/alg-harmony/logs"
        train_writer = tf.compat.v1.summary.FileWriter(LOGDIR)
        train_writer.add_graph(self.session.graph)
        train_writer.flush()
        train_writer.close()

    def sigmoid(self, values, slope, threshold):
        e = tf.exp(-1 * slope * (values - threshold))
        return 1 / (1 + e)

    def get_inhibitions(self):
        # Move each node's inhibition level, towards the level of the energy on that node,
        # proportional to inhibition_rate
        return tf.compat.v1.assign(
            self.inhibitions,
            self.inhibitions
            + (
                tf.pow(self.energies - self.inhibitions, 2)
                * self.inhibition_rate
                * 0.00001
            ),
        )

    def inhibit(self):

        potential_local_averages = tf.einsum(
            "n,nm->m", self.potentials, self.inhibition_averaging_weights
        )

        # Remove potential from a node proportional to its inhibition level and inhibition_gain
        return tf.compat.v1.assign(
            self.potentials,
            tf.clip_by_value(
                self.potentials
                - (
                    self.inhibition_gain
                    * self.inhibitions
                    * self.potentials
                    * potential_local_averages
                ),
                0,
                1,
            ),
        )

    def resonate(self, input, weights, gain, name="resonate"):
        with tf.name_scope(name if self.tensorboard else None):

            resonated = tf.einsum("n,nm->m", input, weights) * gain
            return resonated

    def gravitate(self, attractors, attractees, gravity_coefficients, strength):

        # gravity_coefficients have strength of force from attractee -> attractor ie.
        # (ee1 -> or1) (ee1 -> or2) (ee1 ->or3)
        # (ee2 -> or1) (ee2 -> or2) (ee2 ->or3)
        # (ee3 -> or1) (ee3 -> or2) (ee3 ->or3)

        # Create matrix of forces from each attractor to each attractee.
        # The higher value (larger mass) attractors exert more force
        forces = attractors * gravity_coefficients
        force_row_sums = tf.reduce_sum(forces, 0)

        # Avoid dividing by zero when normalizing rows
        row_normalization_coefficients = tf.where(
            tf.less(force_row_sums, 1e-7), 0.0 * force_row_sums, 1.0 / force_row_sums
        )

        normalized_forces = tf.transpose(
            tf.transpose(forces) * row_normalization_coefficients
        )

        moving_attractees = (
            tf.einsum("n,nm->m", attractees, normalized_forces) * strength
        )
        remaining_attractees = attractees - moving_attractees

        return moving_attractees, remaining_attractees

    def space(self, spacers, spacees, weights, strength):

        neighbourhood_strengths = tf.einsum("n,nm->m", spacers, weights)

        # TODO Remove clip_by_value, spacees should not exceed 1 but do without it for some reason
        return tf.clip_by_value(
            spacees - (spacees * neighbourhood_strengths * 30 * strength), 0, 1
        )


tf.compat.v1.disable_eager_execution()
alg_harmony = AlgHarmony()
alg_harmony.export()
