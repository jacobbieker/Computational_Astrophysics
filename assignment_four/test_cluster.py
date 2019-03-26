from __future__ import division
import numpy
import cluster
from amuse.lab import Particles, nbody_system, constants
from amuse.units import units
from nose.tools import *


def test_saltpeter_cluster_distribution():
    saltpeter_cluster = cluster.cluster(10000, 0.1 | units.MSun, 100 | units.MSun, 3 | units.parsec, 'salpeter')
    assert numpy.isclose(saltpeter_cluster.virial_radius().value_in(units.parsec), 3.0)
    assert len(saltpeter_cluster) == 10000
    assert numpy.min(saltpeter_cluster.mass.value_in(units.MSun)) >= 0.1
    assert numpy.min(saltpeter_cluster.mass.value_in(units.MSun)) <= 100.


def test_miller_scalo_cluster_distribution():
    miller_scalo_cluster = cluster.cluster(10000, 0.1 | units.MSun, 100 | units.MSun, 3 | units.parsec, 'miller_scalo')
    assert numpy.isclose(miller_scalo_cluster.virial_radius().value_in(units.parsec), 3.0)
    assert len(miller_scalo_cluster) == 10000
    assert numpy.min(miller_scalo_cluster.mass.value_in(units.MSun)) >= 0.1
    assert numpy.min(miller_scalo_cluster.mass.value_in(units.MSun)) <= 100.


def test_exp_cluster_distribution():
    exp_cluster = cluster.cluster(10000, 0.1 | units.MSun, 100 | units.MSun, 3 | units.parsec, 'otherexp')
    assert numpy.isclose(exp_cluster.virial_radius().value_in(units.parsec), 3.0)
    assert len(exp_cluster) == 10000
    assert numpy.min(exp_cluster.mass.value_in(units.MSun)) >= 0.1
    assert numpy.min(exp_cluster.mass.value_in(units.MSun)) <= 100.


@raises(NotImplementedError)
def test_none_cluster_distribution():
    cluster.cluster(10000, 0.1 | units.MSun, 100 | units.MSun, 3 | units.parsec, None)


@raises(NotImplementedError)
def test_wrong_string_cluster_distribution():
    cluster.cluster(10000, 0.1 | units.MSun, 100 | units.MSun, 3 | units.parsec, "")


@raises(AttributeError)
def test_too_small_cluster_distribution():
    cluster.cluster(1, 0.1 | units.MSun, 100 | units.MSun, 3 | units.parsec, "otherexp")


def test_large_cluster_distribution():
    large_cluster = cluster.cluster(100000, 0.1 | units.MSun, 100 | units.MSun, 5 | units.parsec, "otherexp")
    assert numpy.isclose(large_cluster.virial_radius().value_in(units.parsec), 5.0)
    assert len(large_cluster) == 100000
    assert numpy.min(large_cluster.mass.value_in(units.MSun)) >= 0.1
    assert numpy.min(large_cluster.mass.value_in(units.MSun)) <= 100.


def test_halfmass():
    test_particles = Particles(1000)
    test_particles.mass = numpy.ones(1000) | units.MSun
    test_particles.x = numpy.linspace(0, 1000, num=1000) | units.parsec
    test_particles.vx = numpy.zeros(1000) | units.parsec / units.s
    test_particles.vy = numpy.zeros(1000) | units.parsec / units.s
    test_particles.vz = numpy.zeros(1000) | units.parsec / units.s
    test_particles.y = numpy.zeros(1000) | units.parsec
    test_particles.z = numpy.zeros(1000) | units.parsec
    halfmass = cluster.halfmass_radius(test_particles)
    assert numpy.isclose(halfmass.value_in(units.parsec), 249.749749749)


def test_dynamical_time():
    test_particles = Particles(1000)
    test_particles.mass = numpy.ones(1000) | units.MSun
    test_particles.x = numpy.linspace(0, 1000, num=1000) | units.parsec
    test_particles.vx = numpy.zeros(1000) | units.parsec / units.s
    test_particles.vy = numpy.zeros(1000) | units.parsec / units.s
    test_particles.vz = numpy.zeros(1000) | units.parsec / units.s
    test_particles.y = numpy.zeros(1000) | units.parsec
    test_particles.z = numpy.zeros(1000) | units.parsec
    dynamical_time = cluster.dynamical_timescale(test_particles)
    assert numpy.isclose(dynamical_time.value_in(units.Myr), 1860.7063959)

