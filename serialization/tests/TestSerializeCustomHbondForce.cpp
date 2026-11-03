/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2010-2014 Stanford University and the Authors.      *
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
#include "openmm/CustomHbondForce.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace OpenMM;
using namespace std;

void testSerialization() {
    // Create a Force.

    CustomHbondForce force("5*sin(x)^2+y*z");
    force.setForceGroup(3);
    force.setName("custom name");
    force.setNonbondedMethod(CustomHbondForce::CutoffPeriodic);
    force.setCutoffDistance(2.1);
    force.addGlobalParameter("x", 1.3);
    force.addGlobalParameter("y", 2.221);
    force.addPerDonorParameter("z");
    force.addPerAcceptorParameter("w");
    force.addPerAcceptorParameter("q");
    vector<double> params(1);
    params[0] = 1.0;
    force.addDonor(0, 1, 2, params);
    params[0] = -3.3;
    force.addDonor(5, 4, 3, params);
    params.resize(2);
    params[0] = 2.1;
    params[1] = 3.3;
    force.addAcceptor(1, 0, -1, params);
    params[0] = -1;
    params[1] = -1.1;
    force.addAcceptor(2, 3, -1, params);
    force.addDonor(6, 7, 8, 9, {2.5});
    force.addAcceptor(9, 8, 7, 6, {0.5, -0.5});
    force.addExclusion(0, 1);
    force.addExclusion(1, 2);
    vector<double> values(10);
    for (int i = 0; i < 10; i++)
        values[i] = sin((double) i);
    force.addTabulatedFunction("f", new Discrete1DFunction(values));

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<CustomHbondForce>(&force, "Force", buffer);
    CustomHbondForce* copy = XmlSerializer::deserialize<CustomHbondForce>(buffer);

    // Compare the two forces to see if they are identical.

    CustomHbondForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getName(), force2.getName());
    ASSERT_EQUAL(force.getEnergyFunction(), force2.getEnergyFunction());
    ASSERT_EQUAL(force.getNonbondedMethod(), force2.getNonbondedMethod());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getNumPerDonorParameters(), force2.getNumPerDonorParameters());
    for (int i = 0; i < force.getNumPerDonorParameters(); i++)
        ASSERT_EQUAL(force.getPerDonorParameterName(i), force2.getPerDonorParameterName(i));
    ASSERT_EQUAL(force.getNumPerAcceptorParameters(), force2.getNumPerAcceptorParameters());
    for (int i = 0; i < force.getNumPerAcceptorParameters(); i++)
        ASSERT_EQUAL(force.getPerAcceptorParameterName(i), force2.getPerAcceptorParameterName(i));
    ASSERT_EQUAL(force.getNumGlobalParameters(), force2.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        ASSERT_EQUAL(force.getGlobalParameterName(i), force2.getGlobalParameterName(i));
        ASSERT_EQUAL(force.getGlobalParameterDefaultValue(i), force2.getGlobalParameterDefaultValue(i));
    }
    ASSERT_EQUAL(force.getNumDonors(), force2.getNumDonors());
    for (int i = 0; i < force.getNumDonors(); i++) {
        int a1, b1, c1, d1, a2, b2, c2, d2;
        vector<double> params1, params2;
        force.getDonorParameters(i, a1, b1, c1, params1);
        force2.getDonorParameters(i, a2, b2, c2, params2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
        ASSERT_EQUAL(c1, c2);
        force.getDonorParticles(i, a1, b1, c1, d1);
        force2.getDonorParticles(i, a2, b2, c2, d2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
        ASSERT_EQUAL(c1, c2);
        ASSERT_EQUAL(d1, d2);
        ASSERT_EQUAL(params1.size(), params2.size());
        for (int j = 0; j < (int) params1.size(); j++)
            ASSERT_EQUAL(params1[j], params2[j]);
    }
    ASSERT_EQUAL(force.getNumAcceptors(), force2.getNumAcceptors());
    for (int i = 0; i < force.getNumAcceptors(); i++) {
        int a1, b1, c1, d1, a2, b2, c2, d2;
        vector<double> params1, params2;
        force.getAcceptorParameters(i, a1, b1, c1, params1);
        force2.getAcceptorParameters(i, a2, b2, c2, params2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
        ASSERT_EQUAL(c1, c2);
        force.getAcceptorParticles(i, a1, b1, c1, d1);
        force2.getAcceptorParticles(i, a2, b2, c2, d2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
        ASSERT_EQUAL(c1, c2);
        ASSERT_EQUAL(d1, d2);
        ASSERT_EQUAL(params1.size(), params2.size());
        for (int j = 0; j < (int) params1.size(); j++)
            ASSERT_EQUAL(params1[j], params2[j]);
    }
    ASSERT_EQUAL(force.getNumExclusions(), force2.getNumExclusions());
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int a1, a2, b1, b2;
        force.getExclusionParticles(i, a1, b1);
        force2.getExclusionParticles(i, a2, b2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
    }
    ASSERT_EQUAL(force.getNumTabulatedFunctions(), force2.getNumTabulatedFunctions());
    for (int i = 0; i < force.getNumTabulatedFunctions(); i++) {
        vector<double> val1, val2;
        dynamic_cast<Discrete1DFunction&>(force.getTabulatedFunction(i)).getFunctionParameters(val1);
        dynamic_cast<Discrete1DFunction&>(force2.getTabulatedFunction(i)).getFunctionParameters(val2);
        ASSERT_EQUAL(force.getTabulatedFunctionName(i), force2.getTabulatedFunctionName(i));
        ASSERT_EQUAL(val1.size(), val2.size());
        for (int j = 0; j < (int) val1.size(); j++)
            ASSERT_EQUAL(val1[j], val2[j]);
    }
}

void testVersion1() {
    // Files written by older versions of OpenMM (version 1) have no fourth particle.
    // Make sure they can still be read.

    CustomHbondForce force("distance(d1,a1)");
    force.addDonor(0, 1, 2);
    force.addAcceptor(3, 4, -1);
    stringstream buffer;
    XmlSerializer::serialize<CustomHbondForce>(&force, "Force", buffer);
    string xml = buffer.str();
    size_t pos = xml.find("version=\"2\"");
    ASSERT(pos != string::npos);
    xml.replace(pos, 11, "version=\"1\"");
    for (pos = xml.find(" p4=\"-1\""); pos != string::npos; pos = xml.find(" p4=\"-1\""))
        xml.erase(pos, 8);
    ASSERT(xml.find("p4") == string::npos);
    stringstream buffer2(xml);
    CustomHbondForce* copy = XmlSerializer::deserialize<CustomHbondForce>(buffer2);
    int p1, p2, p3, p4;
    copy->getDonorParticles(0, p1, p2, p3, p4);
    ASSERT_EQUAL(0, p1);
    ASSERT_EQUAL(1, p2);
    ASSERT_EQUAL(2, p3);
    ASSERT_EQUAL(-1, p4);
    copy->getAcceptorParticles(0, p1, p2, p3, p4);
    ASSERT_EQUAL(3, p1);
    ASSERT_EQUAL(4, p2);
    ASSERT_EQUAL(-1, p3);
    ASSERT_EQUAL(-1, p4);
    delete copy;
}

int main() {
    try {
        testSerialization();
        testVersion1();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
