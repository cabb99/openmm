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
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <utility>
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

/**
 * Compare the forces and energy of two Contexts at a series of random configurations.
 */
void compareContexts(Context& c1, Context& c2, int numParticles, double tol, double scale=2.0) {
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < numParticles; j++)
            positions[j] = Vec3(scale*genrand_real2(sfmt), scale*genrand_real2(sfmt), scale*genrand_real2(sfmt));
        c1.setPositions(positions);
        c2.setPositions(positions);
        State s1 = c1.getState(State::Forces | State::Energy);
        State s2 = c2.getState(State::Forces | State::Energy);
        for (int j = 0; j < numParticles; j++)
            ASSERT_EQUAL_VEC(s2.getForces()[j], s1.getForces()[j], tol);
        ASSERT_EQUAL_TOL(s2.getPotentialEnergy(), s1.getPotentialEnergy(), tol);
    }
}

/**
 * Take a small step along the gradient and check that the energy changes by the amount the forces predict.
 */
void checkForcesMatchEnergy(Context& context, const vector<Vec3>& positions, double tol) {
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    double norm = 0.0;
    for (const Vec3& f : state.getForces())
        norm += f.dot(f);
    norm = sqrt(norm);
    if (norm == 0.0)
        return;
    const double stepSize = 1e-3;
    double step = 0.5*stepSize/norm;
    vector<Vec3> positions2(positions.size()), positions3(positions.size());
    for (int j = 0; j < (int) positions.size(); j++) {
        Vec3 f = state.getForces()[j];
        positions2[j] = positions[j]-f*step;
        positions3[j] = positions[j]+f*step;
    }
    context.setPositions(positions2);
    double energy2 = context.getState(State::Energy).getPotentialEnergy();
    context.setPositions(positions3);
    double energy3 = context.getState(State::Energy).getPotentialEnergy();
    ASSERT_EQUAL_TOL(norm, (energy2-energy3)/stepSize, tol);
}

void testAngle4MatchesAngle() {
    // angle4(p2, p1, p2, p3) is the angle between the vectors p2->p1 and p2->p3, which is angle(p1, p2, p3).

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
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(system1, integrator1, platform);
    Context c2(system2, integrator2, platform);
    compareContexts(c1, c2, numParticles, TOL);
}

void testAngle4() {
    // angle4(p1, p2, p3, p4) is the angle between the vectors p1->p2 and p3->p4, whose particles may
    // come from either group.  Compare to a CustomCompoundBondForce that computes the same angles with
    // pointangle(), and check that the forces are the gradient of the energy.  Particles 0-2 are d1-d3,
    // and particles 3-5 are a1-a3.

    const int numParticles = 6;
    System hbondSystem, compoundSystem;
    for (int i = 0; i < numParticles; i++) {
        hbondSystem.addParticle(1.0);
        compoundSystem.addParticle(1.0);
    }
    CustomHbondForce* hbond = new CustomHbondForce("k*(angle4(d1,d2,a1,a2)-theta0)^2 + sin(angle4(d1,a1,d3,a3)) + 0.5*angle4(a3,d3,d2,a2)^2");
    hbond->addPerDonorParameter("k");
    hbond->addPerAcceptorParameter("theta0");
    hbond->addDonor(0, 1, 2, {1.3});
    hbond->addAcceptor(3, 4, 5, {0.9});
    hbondSystem.addForce(hbond);
    CustomCompoundBondForce* compound = new CustomCompoundBondForce(6,
            "k*(pointangle(x2-x1, y2-y1, z2-z1, 0, 0, 0, x5-x4, y5-y4, z5-z4)-theta0)^2 + "
            "sin(pointangle(x4-x1, y4-y1, z4-z1, 0, 0, 0, x6-x3, y6-y3, z6-z3)) + "
            "0.5*pointangle(x3-x6, y3-y6, z3-z6, 0, 0, 0, x5-x2, y5-y2, z5-z2)^2");
    compound->addPerBondParameter("k");
    compound->addPerBondParameter("theta0");
    compound->addBond({0, 1, 2, 3, 4, 5}, {1.3, 0.9});
    compoundSystem.addForce(compound);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(hbondSystem, integrator1, platform);
    Context c2(compoundSystem, integrator2, platform);
    compareContexts(c1, c2, numParticles, TOL);
    vector<Vec3> positions = {Vec3(0.1, 0.2, 0.3), Vec3(0.5, 0.1, 0.4), Vec3(0.3, 0.9, 0.2), Vec3(1.2, 0.3, 0.1), Vec3(1.4, 0.8, 0.5), Vec3(0.9, 0.2, 0.9)};
    checkForcesMatchEnergy(c1, positions, 1e-3);
}

void testAngle4Orientation() {
    // Reversing either vector replaces the angle by pi minus the angle.  The sum below is therefore
    // constant, so the energy must equal pi and every force must vanish.

    const int numParticles = 6;
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    CustomHbondForce* custom = new CustomHbondForce("0.5*(angle4(d1,d2,a1,a2) + angle4(d2,d1,a1,a2) + angle4(d3,a3,d2,a1) + angle4(d3,a3,a1,d2))");
    custom->addDonor(0, 1, 2);
    custom->addAcceptor(3, 4, 5);
    system.addForce(custom);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < numParticles; j++)
            positions[j] = Vec3(2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt), 2.0*genrand_real2(sfmt));
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        ASSERT_EQUAL_TOL(M_PI, state.getPotentialEnergy(), 1e-5);
        for (int j = 0; j < numParticles; j++)
            ASSERT_EQUAL_VEC(Vec3(), state.getForces()[j], 1e-4);
    }
}

void testAngle4Geometries() {
    // Place the vectors d1->d2 and a1->a2 at chosen angles, including exactly parallel and antiparallel,
    // angles near zero and pi, and angles on both sides of cos(theta) = +-0.99, with very different
    // vector lengths.  The energy angle4(d1,d2,a1,a2) must equal the angle, the forces must be finite and
    // sum to zero, and away from 0 and pi they must equal the analytic gradient.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    CustomHbondForce* custom = new CustomHbondForce("angle4(d1,d2,a1,a2)");
    custom->addDonor(0, 1, -1);
    custom->addAcceptor(2, 3, -1);
    system.addForce(custom);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    double switchAngle = acos(0.99);
    vector<double> angles = {0.0, 1e-4, 1e-2, switchAngle-1e-3, switchAngle+1e-3, M_PI/2, M_PI-switchAngle-1e-3, M_PI-switchAngle+1e-3, M_PI-1e-2, M_PI-1e-4, M_PI};
    vector<pair<double, double> > lengths = {{0.1, 0.1}, {0.01, 10.0}, {10.0, 0.01}};
    Vec3 axis1 = Vec3(1, 2, -0.5)/sqrt(5.25);
    Vec3 axis2 = Vec3(0, 0.25, 1)/sqrt(1.0625);
    axis2 = axis2-axis1*axis1.dot(axis2);
    axis2 = axis2/sqrt(axis2.dot(axis2));
    for (auto& length : lengths) {
        for (double theta : angles) {
            Vec3 u = axis1*length.first;
            Vec3 w = (axis1*cos(theta)+axis2*sin(theta))*length.second;
            vector<Vec3> positions = {Vec3(0.1, 0.2, 0.3), Vec3(0.1, 0.2, 0.3)+u, Vec3(1.1, -0.4, 0.6), Vec3(1.1, -0.4, 0.6)+w};
            context.setPositions(positions);
            State state = context.getState(State::Forces | State::Energy);
            ASSERT_EQUAL_TOL(theta, state.getPotentialEnergy(), 1e-5);
            Vec3 total;
            for (const Vec3& f : state.getForces()) {
                for (int k = 0; k < 3; k++)
                    ASSERT(isfinite(f[k]));
                total += f;
            }
            double maxForce = 1.0/min(length.first, length.second);
            ASSERT_EQUAL_VEC(Vec3(), total/maxForce, 1e-4);
            if (theta < 1e-3 || theta > M_PI-1e-3)
                continue;
            Vec3 uhat = u/length.first, what = w/length.second;
            Vec3 expectedForce2 = (what-uhat*cos(theta))/(length.first*sin(theta));
            Vec3 expectedForce4 = (uhat-what*cos(theta))/(length.second*sin(theta));
            ASSERT_EQUAL_VEC(-expectedForce2, state.getForces()[0], 1e-4);
            ASSERT_EQUAL_VEC(expectedForce2, state.getForces()[1], 1e-4);
            ASSERT_EQUAL_VEC(-expectedForce4, state.getForces()[2], 1e-4);
            ASSERT_EQUAL_VEC(expectedForce4, state.getForces()[3], 1e-4);
        }
    }
}

void testAngle4SharedDeltas() {
    // An expression that combines angle4() with distances, angles, and dihedrals built from the same pairs
    // of particles, in both orders, must still match an equivalent CustomCompoundBondForce.

    const int numParticles = 6;
    System hbondSystem, compoundSystem;
    for (int i = 0; i < numParticles; i++) {
        hbondSystem.addParticle(1.0);
        compoundSystem.addParticle(1.0);
    }
    CustomHbondForce* hbond = new CustomHbondForce("distance(d1,a1) + angle(d2,d1,a1) + cos(dihedral(d2,d1,a1,a2)) + "
            "angle4(d1,a1,d2,a2) + angle4(a1,d1,d2,d1) + 0.5*angle4(d2,d1,a2,a1)^2");
    hbond->addDonor(0, 1, 2);
    hbond->addAcceptor(3, 4, 5);
    hbondSystem.addForce(hbond);
    CustomCompoundBondForce* compound = new CustomCompoundBondForce(6, "distance(p1,p4) + angle(p2,p1,p4) + cos(dihedral(p2,p1,p4,p5)) + "
            "pointangle(x4-x1, y4-y1, z4-z1, 0, 0, 0, x5-x2, y5-y2, z5-z2) + "
            "pointangle(x1-x4, y1-y4, z1-z4, 0, 0, 0, x1-x2, y1-y2, z1-z2) + "
            "0.5*pointangle(x1-x2, y1-y2, z1-z2, 0, 0, 0, x4-x5, y4-y5, z4-z5)^2");
    compound->addBond({0, 1, 2, 3, 4, 5});
    compoundSystem.addForce(compound);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(hbondSystem, integrator1, platform);
    Context c2(compoundSystem, integrator2, platform);
    compareContexts(c1, c2, numParticles, TOL);
}

void testAngle4MatchesBondAngleFromDihedral() {
    // A common way of computing the angle between the d1->d2 and a1->a2 bonds without angle4() combines two
    // angles and a dihedral: cos(theta) = sin(t1)*sin(t2)*cos(phi) - cos(t1)*cos(t2).  angle4(d1,d2,a1,a2)
    // must give the same energy and forces.

    const int numParticles = 6;
    System system1, system2;
    for (int i = 0; i < numParticles; i++) {
        system1.addParticle(1.0);
        system2.addParticle(1.0);
    }
    CustomHbondForce* force1 = new CustomHbondForce("cos(theta)^2+theta; theta=acos(sin(t1)*sin(t2)*cos(phi)-cos(t1)*cos(t2)); "
            "t1=angle(d2,d1,a1); t2=angle(d1,a1,a2); phi=dihedral(d2,d1,a1,a2)");
    CustomHbondForce* force2 = new CustomHbondForce("cos(theta)^2+theta; theta=angle4(d1,d2,a1,a2)");
    for (CustomHbondForce* force : {force1, force2}) {
        force->addDonor(0, 1, -1);
        force->addAcceptor(3, 4, -1);
    }
    system1.addForce(force1);
    system2.addForce(force2);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(system1, integrator1, platform);
    Context c2(system2, integrator2, platform);
    compareContexts(c1, c2, numParticles, 1e-4);
}

void testAngle4Periodic() {
    // Moving particles by periodic box vectors of a triclinic box must not change the result.

    const int numParticles = 6;
    Vec3 a(5, 0, 0), b(1, 5, 0), c(-1, 1.5, 5);
    System system;
    system.setDefaultPeriodicBoxVectors(a, b, c);
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    CustomHbondForce* custom = new CustomHbondForce("angle4(d1,d2,a1,a2) + 2*cos(angle4(d1,a1,d3,a3)) + angle4(d3,a3,d2,a2)^2");
    custom->setNonbondedMethod(CustomHbondForce::CutoffPeriodic);
    custom->setCutoffDistance(2.0);
    custom->addDonor(0, 1, 2);
    custom->addAcceptor(3, 4, 5);
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
    positions[2] += a-b;
    positions[3] -= c;
    positions[5] += a+c;
    context.setPositions(positions);
    State s2 = context.getState(State::Forces | State::Energy);
    for (int j = 0; j < numParticles; j++)
        ASSERT_EQUAL_VEC(s1.getForces()[j], s2.getForces()[j], TOL);
    ASSERT_EQUAL_TOL(s1.getPotentialEnergy(), s2.getPotentialEnergy(), TOL);
}

void testAngle4ManyPairs() {
    // Many donors and acceptors (more than fit in one block on GPUs) with a cutoff that excludes some
    // pairs, plus explicit exclusions.  Compare to a CustomCompoundBondForce with one bond per included pair.

    const int numGroups = 40;
    const double cutoff = 1.0;
    System hbondSystem, compoundSystem;
    CustomHbondForce* hbond = new CustomHbondForce("angle4(d1,d2,a1,a2)^2 + cos(angle4(d2,d3,a3,a1))");
    hbond->setNonbondedMethod(CustomHbondForce::CutoffNonPeriodic);
    hbond->setCutoffDistance(cutoff);
    CustomCompoundBondForce* compound = new CustomCompoundBondForce(6, "step(cutoff-distance(p1,p4))*("
            "pointangle(x2-x1, y2-y1, z2-z1, 0, 0, 0, x5-x4, y5-y4, z5-z4)^2 + "
            "cos(pointangle(x3-x2, y3-y2, z3-z2, 0, 0, 0, x4-x6, y4-y6, z4-z6)))");
    compound->addGlobalParameter("cutoff", cutoff);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions;
    vector<vector<int> > groups;
    for (int i = 0; i < 2*numGroups; i++) {
        Vec3 center(3*genrand_real2(sfmt), 3*genrand_real2(sfmt), 3*genrand_real2(sfmt));
        vector<int> group;
        for (int j = 0; j < 3; j++) {
            group.push_back(positions.size());
            positions.push_back(center + (j == 0 ? Vec3() : Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*0.3));
            hbondSystem.addParticle(1.0);
            compoundSystem.addParticle(1.0);
        }
        groups.push_back(group);
        if (i < numGroups)
            hbond->addDonor(group[0], group[1], group[2]);
        else
            hbond->addAcceptor(group[0], group[1], group[2]);
    }
    set<pair<int, int> > excluded;
    for (int i = 0; i < numGroups; i += 4) {
        hbond->addExclusion(i, (3*i+1)%numGroups);
        excluded.insert(make_pair(i, (3*i+1)%numGroups));
    }
    for (int i = 0; i < numGroups; i++)
        for (int j = 0; j < numGroups; j++)
            if (excluded.find(make_pair(i, j)) == excluded.end()) {
                vector<int> particles = groups[i];
                particles.insert(particles.end(), groups[numGroups+j].begin(), groups[numGroups+j].end());
                compound->addBond(particles);
            }
    hbondSystem.addForce(hbond);
    compoundSystem.addForce(compound);
    VerletIntegrator integrator1(0.01);
    VerletIntegrator integrator2(0.01);
    Context c1(hbondSystem, integrator1, platform);
    Context c2(compoundSystem, integrator2, platform);
    c1.setPositions(positions);
    c2.setPositions(positions);
    State s1 = c1.getState(State::Forces | State::Energy);
    State s2 = c2.getState(State::Forces | State::Energy);
    for (int i = 0; i < (int) positions.size(); i++)
        ASSERT_EQUAL_VEC(s2.getForces()[i], s1.getForces()[i], 1e-4);
    ASSERT_EQUAL_TOL(s2.getPotentialEnergy(), s1.getPotentialEnergy(), 1e-4);
}

void testAngle4IllegalArguments() {
    // Each vector in angle4() needs two different particles, and angle4() takes exactly four of them.

    for (string expression : {"angle4(d1,d1,a1,a2)", "angle4(d1,d2,a2,a2)", "angle4(d1,d2,a1)"}) {
        System system;
        for (int i = 0; i < 4; i++)
            system.addParticle(1.0);
        CustomHbondForce* custom = new CustomHbondForce(expression);
        custom->addDonor(0, 1, -1);
        custom->addAcceptor(2, 3, -1);
        system.addForce(custom);
        VerletIntegrator integrator(0.01);
        bool threwException = false;
        try {
            Context context(system, integrator, platform);
        }
        catch (const exception& e) {
            threwException = true;
        }
        ASSERT(threwException);
    }
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
        testAngle4MatchesAngle();
        testAngle4();
        testAngle4Orientation();
        testAngle4Geometries();
        testAngle4SharedDeltas();
        testAngle4MatchesBondAngleFromDihedral();
        testAngle4Periodic();
        testAngle4ManyPairs();
        testAngle4IllegalArguments();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}

