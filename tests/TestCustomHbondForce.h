/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2018 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/CustomCompoundBondForce.h"
#include "openmm/CustomHbondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "sfmt/SFMT.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

void testHbond() {
    // Create a system using a CustomHbondForce.

    System customSystem;
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    customSystem.addParticle(1.0);
    CustomHbondForce* custom = new CustomHbondForce("0.5*kr*(distance(d1,a1)-r0)^2 + 0.5*ktheta*(angle(a1,d1,d2)-theta0)^2 + 0.5*kpsi*(angle(d1,a1,a2)-psi0)^2 + kchi*(1+cos(n*dihedral(a3,a2,a1,d1)-chi0))");
    custom->addPerDonorParameter("r0");
    custom->addPerDonorParameter("theta0");
    custom->addPerDonorParameter("psi0");
    custom->addPerAcceptorParameter("chi0");
    custom->addPerAcceptorParameter("n");
    custom->addGlobalParameter("kr", 0.4);
    custom->addGlobalParameter("ktheta", 0.5);
    custom->addGlobalParameter("kpsi", 0.6);
    custom->addGlobalParameter("kchi", 0.7);
    vector<double> parameters(3);
    parameters[0] = 1.5;
    parameters[1] = 1.7;
    parameters[2] = 1.9;
    custom->addDonor(1, 0, -1, parameters);
    parameters.resize(2);
    parameters[0] = 2.1;
    parameters[1] = 2;
    custom->addAcceptor(2, 3, 4, parameters);
    custom->setCutoffDistance(10.0);
    customSystem.addForce(custom);
    ASSERT(!custom->usesPeriodicBoundaryConditions());
    ASSERT(!customSystem.usesPeriodicBoundaryConditions());

    // Create an identical system using HarmonicBondForce, HarmonicAngleForce, and PeriodicTorsionForce.

    System standardSystem;
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    standardSystem.addParticle(1.0);
    HarmonicBondForce* bond = new HarmonicBondForce();
    bond->addBond(1, 2, 1.5, 0.4);
    standardSystem.addForce(bond);
    HarmonicAngleForce* angle = new HarmonicAngleForce();
    angle->addAngle(0, 1, 2, 1.7, 0.5);
    angle->addAngle(1, 2, 3, 1.9, 0.6);
    standardSystem.addForce(angle);
    PeriodicTorsionForce* torsion = new PeriodicTorsionForce();
    torsion->addTorsion(1, 2, 3, 4, 2, 2.1, 0.7);
    standardSystem.addForce(torsion);

    // Set the atoms in various positions, and verify that both systems give identical forces and energy.

    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    vector<Vec3> positions(5);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(customSystem, integrator1, platform);
    Context c2(standardSystem, integrator2, platform);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < (int) positions.size(); j++)
            positions[j] = Vec3(2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt));
        c1.setPositions(positions);
        c2.setPositions(positions);
        State s1 = c1.getState(State::Forces | State::Energy);
        State s2 = c2.getState(State::Forces | State::Energy);
        for (int i = 0; i < customSystem.getNumParticles(); i++)
            ASSERT_EQUAL_VEC(s2.getForces()[i], s1.getForces()[i], TOL);
        ASSERT_EQUAL_TOL(s2.getPotentialEnergy(), s1.getPotentialEnergy(), TOL);
    }
    
    // Try changing the parameters and make sure it's still correct.
    
    parameters.resize(3);
    parameters[0] = 1.4;
    parameters[1] = 1.7;
    parameters[2] = 1.9;
    custom->setDonorParameters(0, 1, 0, -1, parameters);
    parameters.resize(2);
    parameters[0] = 2.2;
    parameters[1] = 2;
    custom->setAcceptorParameters(0, 2, 3, 4, parameters);
    bond->setBondParameters(0, 1, 2, 1.4, 0.4);
    torsion->setTorsionParameters(0, 1, 2, 3, 4, 2, 2.2, 0.7);
    custom->updateParametersInContext(c1);
    bond->updateParametersInContext(c2);
    torsion->updateParametersInContext(c2);
    State s1 = c1.getState(State::Forces | State::Energy);
    State s2 = c2.getState(State::Forces | State::Energy);
    for (int i = 0; i < customSystem.getNumParticles(); i++)
        ASSERT_EQUAL_VEC(s2.getForces()[i], s1.getForces()[i], TOL);
    ASSERT_EQUAL_TOL(s2.getPotentialEnergy(), s1.getPotentialEnergy(), TOL);
}

void testExclusions() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomHbondForce* custom = new CustomHbondForce("(distance(d1,a1)-1)^2");
    custom->addDonor(0, 1, -1, vector<double>());
    custom->addDonor(1, 0, -1, vector<double>());
    custom->addAcceptor(2, 0, -1, vector<double>());
    custom->addExclusion(1, 0);
    system.addForce(custom);
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 2, 0);
    positions[2] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    ASSERT_EQUAL_VEC(Vec3(2, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(-2, 0, 0), forces[2], TOL);
    ASSERT_EQUAL_TOL(1.0, state.getPotentialEnergy(), TOL);
}

void testCutoff() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomHbondForce* custom = new CustomHbondForce("(distance(d1,a1)-1)^2");
    custom->addDonor(0, 1, -1, vector<double>());
    custom->addDonor(1, 0, -1, vector<double>());
    custom->addAcceptor(2, 0, -1, vector<double>());
    custom->setNonbondedMethod(CustomHbondForce::CutoffNonPeriodic);
    custom->setCutoffDistance(2.5);
    system.addForce(custom);
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 3, 0);
    positions[2] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    ASSERT_EQUAL_VEC(Vec3(2, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(-2, 0, 0), forces[2], TOL);
    ASSERT_EQUAL_TOL(1.0, state.getPotentialEnergy(), TOL);
}

void testCustomFunctions() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomHbondForce* custom = new CustomHbondForce("foo(distance(d1,a1))");
    custom->addDonor(1, 0, -1, vector<double>());
    custom->addDonor(2, 0, -1, vector<double>());
    custom->addAcceptor(0, 1, -1, vector<double>());
    vector<double> function(2);
    function[0] = 0;
    function[1] = 1;
    custom->addTabulatedFunction("foo", new Continuous1DFunction(function, 0, 10));
    system.addForce(custom);
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 2, 0);
    positions[2] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    ASSERT_EQUAL_VEC(Vec3(0.1, 0.1, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, -0.1, 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(-0.1, 0, 0), forces[2], TOL);
    ASSERT_EQUAL_TOL(0.1*2+0.1*2, state.getPotentialEnergy(), TOL);

    // Try updating the tabulated function.

    for (int i = 0; i < function.size(); i++)
        function[i] *= 0.5;
    dynamic_cast<Continuous1DFunction&>(custom->getTabulatedFunction(0)).setFunctionParameters(function, 0, 10);
    custom->updateParametersInContext(context);
    state = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(0.5*(0.1*2+0.1*2), state.getPotentialEnergy(), TOL);
}

void test2DFunction() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomHbondForce* custom = new CustomHbondForce("tab(dtype, atype)");
    custom->addPerDonorParameter("dtype");
    custom->addPerAcceptorParameter("atype");
    custom->addDonor(1, 0, -1, {0.0});
    custom->addDonor(2, 0, -1, {2.0});
    custom->addAcceptor(0, 1, -1, {1.0});
    vector<double> function = {0.0, 1.0, 2.0, 5.0, 6.0, 7.0};
    custom->addTabulatedFunction("tab", new Discrete2DFunction(3, 2, function));
    system.addForce(custom);
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 2, 0);
    positions[2] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[2], TOL);
    ASSERT_EQUAL_TOL(12.0, state.getPotentialEnergy(), TOL);
}

void testIllegalVariable() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    CustomHbondForce* force = new CustomHbondForce("1+none");
    force->addDonor(0, -1, -1);
    force->addAcceptor(1, -1, -1);
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    bool threwException = false;
    try {
        Context(system, integrator, platform);
    }
    catch (const exception& e) {
        threwException = true;
    }
    ASSERT(threwException);
}

void testParameters() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    CustomHbondForce* custom = new CustomHbondForce("(2*d+a)*distance(d1,a1)");
    custom->addPerDonorParameter("d");
    custom->addPerAcceptorParameter("a");
    custom->addDonor(1, 0, -1, vector<double>({1.5}));
    custom->addDonor(2, 0, -1, vector<double>({1.8}));
    custom->addAcceptor(0, 1, -1, vector<double>({2.1}));
    system.addForce(custom);
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 2, 0);
    positions[2] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    ASSERT_EQUAL_VEC(Vec3((2*1.8+2.1), (2*1.5+2.1), 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, -(2*1.5+2.1), 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(-(2*1.8+2.1), 0, 0), forces[2], TOL);
    ASSERT_EQUAL_TOL(2*(2*1.8+2.1)+2*(2*1.5+2.1), state.getPotentialEnergy(), TOL);
}

void testLargeSystem(CustomHbondForce::NonbondedMethod method) {
    int numParticles = 5000;
    double boxSize = 3.0;
    double cutoff = 1.0;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    CustomHbondForce* custom = new CustomHbondForce("(distance(d1,a1)-1)^2");
    custom->setNonbondedMethod(method);
    custom->setCutoffDistance(cutoff);
    NonbondedForce* nb = new NonbondedForce(); // So that atom reordering will be done
    nb->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        if (i%2 == 0)
            custom->addDonor(i, -1, -1);
        else
            custom->addAcceptor(i, -1, -1);
        positions[i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
        nb->addParticle(0.0, 1.0, 0.0);
    }
    system.addForce(custom);
    system.addForce(nb);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    double expectedEnergy = 0;
    for (int i = 0; i < numParticles; i += 2) {
        Vec3 expectedForce;
        for (int j = 1; j < numParticles; j += 2) {
            Vec3 d = positions[i]-positions[j];
            if (method == CustomHbondForce::CutoffPeriodic) {
                d[0] -= round(d[0]/boxSize)*boxSize;
                d[1] -= round(d[1]/boxSize)*boxSize;
                d[2] -= round(d[2]/boxSize)*boxSize;
            }
            double r = sqrt(d.dot(d));
            if (method == CustomHbondForce::NoCutoff || r < cutoff) {
                expectedEnergy += (r-1)*(r-1);
                expectedForce -= 2*(r-1)*d/r;
            }
        }
        ASSERT_EQUAL_VEC(expectedForce, state.getForces()[i], 1e-5);
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testFourParticleGroups() {
    // Create a system using a CustomHbondForce whose donor and acceptor groups each have four particles.
    // Particles 0-3 are d1-d4, and particles 4-7 are a1-a4.

    const int numParticles = 8;
    System customSystem;
    for (int i = 0; i < numParticles; i++)
        customSystem.addParticle(1.0);
    CustomHbondForce* custom = new CustomHbondForce("0.5*kr*(distance(d1,a1)-r0)^2 + 0.5*kr*(distance(d3,a4)-r1)^2 + "
            "0.5*ktheta*(angle(d4,d1,a1)-theta0)^2 + 0.5*ktheta*(angle(d1,a1,a4)-theta1)^2 + kchi*(1+cos(dihedral(d4,d1,a1,a4)-chi0))");
    custom->addPerDonorParameter("r0");
    custom->addPerDonorParameter("theta0");
    custom->addPerAcceptorParameter("r1");
    custom->addPerAcceptorParameter("theta1");
    custom->addPerAcceptorParameter("chi0");
    custom->addGlobalParameter("kr", 0.4);
    custom->addGlobalParameter("ktheta", 0.5);
    custom->addGlobalParameter("kchi", 0.7);
    custom->addDonor(0, 1, 2, 3, {1.5, 1.7});
    custom->addAcceptor(4, 5, 6, 7, {1.1, 1.9, 2.1});
    customSystem.addForce(custom);

    // Check that the particles were recorded correctly.

    int p1, p2, p3, p4;
    vector<double> params;
    custom->getDonorParticles(0, p1, p2, p3, p4);
    ASSERT_EQUAL(0, p1);
    ASSERT_EQUAL(1, p2);
    ASSERT_EQUAL(2, p3);
    ASSERT_EQUAL(3, p4);
    custom->getAcceptorParticles(0, p1, p2, p3, p4);
    ASSERT_EQUAL(4, p1);
    ASSERT_EQUAL(5, p2);
    ASSERT_EQUAL(6, p3);
    ASSERT_EQUAL(7, p4);
    custom->getDonorParameters(0, p1, p2, p3, params);
    ASSERT_EQUAL(0, p1);
    ASSERT_EQUAL(1, p2);
    ASSERT_EQUAL(2, p3);
    ASSERT_EQUAL(2, params.size());
    ASSERT_EQUAL(1.5, params[0]);

    // Create an identical system using HarmonicBondForce, HarmonicAngleForce, and PeriodicTorsionForce.

    System standardSystem;
    for (int i = 0; i < numParticles; i++)
        standardSystem.addParticle(1.0);
    HarmonicBondForce* bond = new HarmonicBondForce();
    bond->addBond(0, 4, 1.5, 0.4);
    bond->addBond(2, 7, 1.1, 0.4);
    standardSystem.addForce(bond);
    HarmonicAngleForce* angle = new HarmonicAngleForce();
    angle->addAngle(3, 0, 4, 1.7, 0.5);
    angle->addAngle(0, 4, 7, 1.9, 0.5);
    standardSystem.addForce(angle);
    PeriodicTorsionForce* torsion = new PeriodicTorsionForce();
    torsion->addTorsion(3, 0, 4, 7, 1, 2.1, 0.7);
    standardSystem.addForce(torsion);

    // Set the atoms in various positions, and verify that both systems give identical forces and energy.

    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(customSystem, integrator1, platform);
    Context c2(standardSystem, integrator2, platform);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < numParticles; j++)
            positions[j] = Vec3(2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt));
        c1.setPositions(positions);
        c2.setPositions(positions);
        State s1 = c1.getState(State::Forces | State::Energy);
        State s2 = c2.getState(State::Forces | State::Energy);
        for (int j = 0; j < numParticles; j++)
            ASSERT_EQUAL_VEC(s2.getForces()[j], s1.getForces()[j], TOL);
        ASSERT_EQUAL_TOL(s2.getPotentialEnergy(), s1.getPotentialEnergy(), TOL);
    }

    // Change the parameters and make sure it's still correct.

    custom->setDonorParameters(0, 0, 1, 2, 3, {1.4, 1.6});
    custom->setAcceptorParameters(0, 4, 5, 6, 7, {1.2, 1.9, 2.2});
    bond->setBondParameters(0, 0, 4, 1.4, 0.4);
    bond->setBondParameters(1, 2, 7, 1.2, 0.4);
    angle->setAngleParameters(0, 3, 0, 4, 1.6, 0.5);
    torsion->setTorsionParameters(0, 3, 0, 4, 7, 1, 2.2, 0.7);
    custom->updateParametersInContext(c1);
    bond->updateParametersInContext(c2);
    angle->updateParametersInContext(c2);
    torsion->updateParametersInContext(c2);
    State s1 = c1.getState(State::Forces | State::Energy);
    State s2 = c2.getState(State::Forces | State::Energy);
    for (int j = 0; j < numParticles; j++)
        ASSERT_EQUAL_VEC(s2.getForces()[j], s1.getForces()[j], TOL);
    ASSERT_EQUAL_TOL(s2.getPotentialEnergy(), s1.getPotentialEnergy(), TOL);

    // Setting a group with the three particle version of the method removes the fourth particle.

    custom->setAcceptorParameters(0, 4, 5, 6, {1.2, 1.9, 2.2});
    custom->getAcceptorParticles(0, p1, p2, p3, p4);
    ASSERT_EQUAL(-1, p4);

    // An illegal index for the fourth particle should be detected.

    custom->setDonorParameters(0, 0, 1, 2, numParticles, {1.4, 1.6});
    VerletIntegrator integrator3(0.01);
    bool threwException = false;
    try {
        Context(customSystem, integrator3, platform);
    }
    catch (const exception& e) {
        threwException = true;
    }
    ASSERT(threwException);
}

void testFourParticleGroupsPeriodic() {
    // Put the fourth particle of each group in a different periodic copy and make sure
    // the result is unchanged.

    const int numParticles = 8;
    const double boxSize = 5.0;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    CustomHbondForce* custom = new CustomHbondForce("distance(d1,a1)^2 + distance(d4,a4)^2 + angle(d4,d1,a1) + angle(d1,a1,a4) + cos(dihedral(d4,d1,a1,a4))");
    custom->setNonbondedMethod(CustomHbondForce::CutoffPeriodic);
    custom->setCutoffDistance(2.0);
    custom->addDonor(0, 1, 2, 3, {});
    custom->addAcceptor(4, 5, 6, 7, {});
    system.addForce(custom);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    for (int j = 0; j < numParticles; j++)
        positions[j] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt));
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State s1 = context.getState(State::Forces | State::Energy);
    positions[3] += Vec3(boxSize, -boxSize, 0);
    positions[7] += Vec3(0, 2*boxSize, -boxSize);
    context.setPositions(positions);
    State s2 = context.getState(State::Forces | State::Energy);
    for (int j = 0; j < numParticles; j++)
        ASSERT_EQUAL_VEC(s1.getForces()[j], s2.getForces()[j], TOL);
    ASSERT_EQUAL_TOL(s1.getPotentialEnergy(), s2.getPotentialEnergy(), TOL);
}

void testAngle4MatchesAngle() {
    // angle4(p2, p1, p2, p3) should be identical to angle(p1, p2, p3).

    const int numParticles = 6;
    System system1, system2;
    for (int i = 0; i < numParticles; i++) {
        system1.addParticle(1.0);
        system2.addParticle(1.0);
    }
    CustomHbondForce* force1 = new CustomHbondForce("k*(angle(d2,d1,a1)-theta0)^2 + cos(angle(a2,a1,d1))");
    CustomHbondForce* force2 = new CustomHbondForce("k*(angle4(d1,d2,d1,a1)-theta0)^2 + cos(angle4(a1,a2,a1,d1))");
    for (CustomHbondForce* force : {force1, force2}) {
        force->addGlobalParameter("k", 1.3);
        force->addGlobalParameter("theta0", 1.8);
        force->addDonor(0, 1, 2);
        force->addAcceptor(3, 4, 5);
    }
    system1.addForce(force1);
    system2.addForce(force2);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(system1, integrator1, platform);
    Context c2(system2, integrator2, platform);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < numParticles; j++)
            positions[j] = Vec3(2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt));
        c1.setPositions(positions);
        c2.setPositions(positions);
        State s1 = c1.getState(State::Forces | State::Energy);
        State s2 = c2.getState(State::Forces | State::Energy);
        for (int j = 0; j < numParticles; j++)
            ASSERT_EQUAL_VEC(s1.getForces()[j], s2.getForces()[j], TOL);
        ASSERT_EQUAL_TOL(s1.getPotentialEnergy(), s2.getPotentialEnergy(), TOL);
    }
}

void testAngle4() {
    // Compare angle4() to an equivalent CustomCompoundBondForce that computes the same angles with pointangle().
    // Particles 0-3 are d1-d4, and particles 4-7 are a1-a4.

    const int numParticles = 8;
    System hbondSystem, compoundSystem;
    for (int i = 0; i < numParticles; i++) {
        hbondSystem.addParticle(1.0);
        compoundSystem.addParticle(1.0);
    }
    CustomHbondForce* hbond = new CustomHbondForce("k*(angle4(d1,d2,a1,a2)-theta0)^2 + sin(angle4(d1,a1,d4,a4)) + 0.5*angle4(a3,d3,d2,a2)^2");
    hbond->addPerDonorParameter("k");
    hbond->addPerAcceptorParameter("theta0");
    hbond->addDonor(0, 1, 2, 3, {1.3});
    hbond->addAcceptor(4, 5, 6, 7, {0.9});
    hbondSystem.addForce(hbond);
    CustomCompoundBondForce* compound = new CustomCompoundBondForce(8,
            "k*(pointangle(x2-x1, y2-y1, z2-z1, 0, 0, 0, x6-x5, y6-y5, z6-z5)-theta0)^2 + "
            "sin(pointangle(x5-x1, y5-y1, z5-z1, 0, 0, 0, x8-x4, y8-y4, z8-z4)) + "
            "0.5*pointangle(x3-x7, y3-y7, z3-z7, 0, 0, 0, x6-x2, y6-y2, z6-z2)^2");
    compound->addPerBondParameter("k");
    compound->addPerBondParameter("theta0");
    compound->addBond({0, 1, 2, 3, 4, 5, 6, 7}, {1.3, 0.9});
    compoundSystem.addForce(compound);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(hbondSystem, integrator1, platform);
    Context c2(compoundSystem, integrator2, platform);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < numParticles; j++)
            positions[j] = Vec3(2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt));
        c1.setPositions(positions);
        c2.setPositions(positions);
        State s1 = c1.getState(State::Forces | State::Energy);
        State s2 = c2.getState(State::Forces | State::Energy);
        for (int j = 0; j < numParticles; j++)
            ASSERT_EQUAL_VEC(s2.getForces()[j], s1.getForces()[j], TOL);
        ASSERT_EQUAL_TOL(s2.getPotentialEnergy(), s1.getPotentialEnergy(), TOL);
    }

    // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.

    c1.setPositions(positions);
    State state = c1.getState(State::Forces | State::Energy);
    double norm = 0.0;
    for (const Vec3& f : state.getForces())
        norm += f.dot(f);
    norm = sqrt(norm);
    const double stepSize = 1e-3;
    double step = 0.5*stepSize/norm;
    vector<Vec3> positions2(numParticles), positions3(numParticles);
    for (int j = 0; j < numParticles; j++) {
        Vec3 f = state.getForces()[j];
        positions2[j] = positions[j]-f*step;
        positions3[j] = positions[j]+f*step;
    }
    c1.setPositions(positions2);
    double energy2 = c1.getState(State::Energy).getPotentialEnergy();
    c1.setPositions(positions3);
    double energy3 = c1.getState(State::Energy).getPotentialEnergy();
    ASSERT_EQUAL_TOL(norm, (energy2-energy3)/stepSize, 1e-3);
}

void testAngle4Periodic() {
    // Put particles in different periodic copies of a triclinic box and make sure the result is unchanged.

    const int numParticles = 8;
    Vec3 a(5, 0, 0), b(1, 5, 0), c(-1, 1.5, 5);
    System system;
    system.setDefaultPeriodicBoxVectors(a, b, c);
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    CustomHbondForce* custom = new CustomHbondForce("angle4(d1,d2,a1,a2) + 2*cos(angle4(d1,a1,d4,a4)) + angle4(d3,a3,d2,a2)^2");
    custom->setNonbondedMethod(CustomHbondForce::CutoffPeriodic);
    custom->setCutoffDistance(2.0);
    custom->addDonor(0, 1, 2, 3, {});
    custom->addAcceptor(4, 5, 6, 7, {});
    system.addForce(custom);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    for (int j = 0; j < numParticles; j++)
        positions[j] = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt));
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State s1 = context.getState(State::Forces | State::Energy);
    positions[1] += b*2;
    positions[3] += a-b;
    positions[4] -= c;
    positions[7] += a+c;
    context.setPositions(positions);
    State s2 = context.getState(State::Forces | State::Energy);
    for (int j = 0; j < numParticles; j++)
        ASSERT_EQUAL_VEC(s1.getForces()[j], s2.getForces()[j], TOL);
    ASSERT_EQUAL_TOL(s1.getPotentialEnergy(), s2.getPotentialEnergy(), TOL);
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testHbond();
        testExclusions();
        testCutoff();
        testCustomFunctions();
        test2DFunction();
        testIllegalVariable();
        testParameters();
        testLargeSystem(CustomHbondForce::NoCutoff);
        testLargeSystem(CustomHbondForce::CutoffNonPeriodic);
        testLargeSystem(CustomHbondForce::CutoffPeriodic);
        testFourParticleGroups();
        testFourParticleGroupsPeriodic();
        testAngle4MatchesAngle();
        testAngle4();
        testAngle4Periodic();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}

