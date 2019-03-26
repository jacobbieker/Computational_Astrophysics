from __future__ import division
import numpy
import cluster
from amuse.lab import Particles, nbody_system, constants
from amuse.units import units
from nose.tools import *


def test_saltpeter_cluster_distribution():
    saltpeter_cluster = cluster.cluster(10000, 0.1, 100, 3, 'saltpeter')
    assert numpy.isclose(saltpeter_cluster.virial_radius().value_in(units.parsec), 3.0)
    assert len(saltpeter_cluster.particles) == 10000
    assert numpy.min(saltpeter_cluster.particles.mass.value_in(units.MSun)) >= 0.1
    assert numpy.min(saltpeter_cluster.particles.mass.value_in(units.MSun)) <= 100.


def test_miller_scalo_cluster_distribution():
    miller_scalo_cluster = cluster.cluster(10000, 0.1, 100, 3, 'miller_scalo')
    assert numpy.isclose(miller_scalo_cluster.virial_radius().value_in(units.parsec), 3.0)
    assert len(miller_scalo_cluster.particles) == 10000
    assert numpy.min(miller_scalo_cluster.particles.mass.value_in(units.MSun)) >= 0.1
    assert numpy.min(miller_scalo_cluster.particles.mass.value_in(units.MSun)) <= 100.


def test_exp_cluster_distribution():
    exp_cluster = cluster.cluster(10000, 0.1, 100, 3, 'otherexp')
    assert numpy.isclose(exp_cluster.virial_radius().value_in(units.parsec), 3.0)
    assert len(exp_cluster.particles) == 10000
    assert numpy.min(exp_cluster.particles.mass.value_in(units.MSun)) >= 0.1
    assert numpy.min(exp_cluster.particles.mass.value_in(units.MSun)) <= 100.


@raises(NotImplementedError)
def test_none_cluster_distribution():
    cluster.cluster(10000, 0.1, 100, 3, None)


@raises(NotImplementedError)
def test_wrong_string_cluster_distribution():
    cluster.cluster(10000, 0.1, 100, 3, "")


@raises(NotImplementedError)
def test_too_small_cluster_distribution():
    cluster.cluster(1, 0.1, 100, 3, "otherexp")


def test_large_cluster_distribution():
    large_cluster = cluster.cluster(1000000, 0.1, 100, 5, "otherexp")
    assert numpy.isclose(large_cluster.virial_radius().value_in(units.parsec), 5.0)
    assert len(large_cluster.particles) == 1000000
    assert numpy.min(large_cluster.particles.mass.value_in(units.MSun)) >= 0.1
    assert numpy.min(large_cluster.particles.mass.value_in(units.MSun)) <= 100.


def test_animation():
    raise NotImplementedError


def test_dynamical_time():
    saltpeter_cluster = cluster.cluster(10000, 0.1, 100, 3, 'saltpeter')
    dynamical_time = cluster.dynamical_timescale(cluster.halfmass_radius(saltpeter_cluster), saltpeter_cluster)
    raise NotImplementedError


def test_halfmass():
    raise NotImplementedError
