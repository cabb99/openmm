/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2014 Stanford University and the Authors.      *
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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/CustomResiduePairForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/CustomResiduePairForceImpl.h"
#include <cmath>
#include <map>
#include <sstream>
#include <utility>

using namespace OpenMM;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

CustomResiduePairForce::CustomResiduePairForce(const string& energy) : energyExpression(energy), nonbondedMethod(NoCutoff), cutoffDistance(1.0) {
}


CustomResiduePairForce::~CustomResiduePairForce() {
    for (auto function : functions)
        delete function.function;
}

const string& CustomResiduePairForce::getEnergyFunction() const {
    return energyExpression;
}

void CustomResiduePairForce::setEnergyFunction(const std::string& energy) {
    energyExpression = energy;
}

CustomResiduePairForce::NonbondedMethod CustomResiduePairForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void CustomResiduePairForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 2)
        throw OpenMMException("CustomResiduePairForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double CustomResiduePairForce::getCutoffDistance() const {
    return cutoffDistance;
}

void CustomResiduePairForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

int CustomResiduePairForce::addPerDonorParameter(const string& name) {
    donorParameters.push_back(PerPairParameterInfo(name));
    return donorParameters.size()-1;
}

const string& CustomResiduePairForce::getPerDonorParameterName(int index) const {
    ASSERT_VALID_INDEX(index, donorParameters);
    return donorParameters[index].name;
}

void CustomResiduePairForce::setPerDonorParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, donorParameters);
    donorParameters[index].name = name;
}

int CustomResiduePairForce::addPerAcceptorParameter(const string& name) {
    acceptorParameters.push_back(PerPairParameterInfo(name));
    return acceptorParameters.size()-1;
}

const string& CustomResiduePairForce::getPerAcceptorParameterName(int index) const {
    ASSERT_VALID_INDEX(index, acceptorParameters);
    return acceptorParameters[index].name;
}

void CustomResiduePairForce::setPerAcceptorParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, acceptorParameters);
    acceptorParameters[index].name = name;
}

int CustomResiduePairForce::addGlobalParameter(const string& name, double defaultValue) {
    globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
    return globalParameters.size()-1;
}

const string& CustomResiduePairForce::getGlobalParameterName(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].name;
}

void CustomResiduePairForce::setGlobalParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].name = name;
}

double CustomResiduePairForce::getGlobalParameterDefaultValue(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].defaultValue;
}

void CustomResiduePairForce::setGlobalParameterDefaultValue(int index, double defaultValue) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].defaultValue = defaultValue;
}

int CustomResiduePairForce::addDonor(int d1, int d2, int d3, int d4, const vector<double>& parameters) {
    donors.push_back(GroupInfo(d1, d2, d3, d4, parameters));
    return donors.size()-1;
}

void CustomResiduePairForce::getDonorParameters(int index, int& d1, int& d2, int&  d3, int&  d4, std::vector<double>& parameters) const {
    ASSERT_VALID_INDEX(index, donors);
    d1 = donors[index].p1;
    d2 = donors[index].p2;
    d3 = donors[index].p3;
    d4 = donors[index].p4;
    parameters = donors[index].parameters;
}

void CustomResiduePairForce::setDonorParameters(int index, int d1, int d2, int d3, int d4, const vector<double>& parameters) {
    ASSERT_VALID_INDEX(index, donors);
    donors[index].p1 = d1;
    donors[index].p2 = d2;
    donors[index].p3 = d3;
    donors[index].p4 = d4;
    donors[index].parameters = parameters;
}

int CustomResiduePairForce::addAcceptor(int a1, int a2, int a3, int a4, const vector<double>& parameters) {
    acceptors.push_back(GroupInfo(a1, a2, a3, a4, parameters));
    return acceptors.size()-1;
}

void CustomResiduePairForce::getAcceptorParameters(int index, int& a1, int& a2, int& a3, int& a4, std::vector<double>& parameters) const {
    ASSERT_VALID_INDEX(index, acceptors);
    a1 = acceptors[index].p1;
    a2 = acceptors[index].p2;
    a3 = acceptors[index].p3;
    a4 = acceptors[index].p4;
    parameters = acceptors[index].parameters;
}

void CustomResiduePairForce::setAcceptorParameters(int index, int a1, int a2, int a3, int a4, const vector<double>& parameters) {
    ASSERT_VALID_INDEX(index, acceptors);
    acceptors[index].p1 = a1;
    acceptors[index].p2 = a2;
    acceptors[index].p3 = a3;
    acceptors[index].p4 = a4;
    acceptors[index].parameters = parameters;
}

int CustomResiduePairForce::addExclusion(int donor, int acceptor) {
    exclusions.push_back(ExclusionInfo(donor, acceptor));
    return exclusions.size()-1;
}
void CustomResiduePairForce::getExclusionParticles(int index, int& donor, int& acceptor) const {
    ASSERT_VALID_INDEX(index, exclusions);
    donor = exclusions[index].donor;
    acceptor = exclusions[index].acceptor;
}

void CustomResiduePairForce::setExclusionParticles(int index, int donor, int acceptor) {
    ASSERT_VALID_INDEX(index, exclusions);
    exclusions[index].donor = donor;
    exclusions[index].acceptor = acceptor;
}

int CustomResiduePairForce::addTabulatedFunction(const std::string& name, TabulatedFunction* function) {
    functions.push_back(FunctionInfo(name, function));
    return functions.size()-1;
}

const TabulatedFunction& CustomResiduePairForce::getTabulatedFunction(int index) const {
    ASSERT_VALID_INDEX(index, functions);
    return *functions[index].function;
}

TabulatedFunction& CustomResiduePairForce::getTabulatedFunction(int index) {
    ASSERT_VALID_INDEX(index, functions);
    return *functions[index].function;
}

const string& CustomResiduePairForce::getTabulatedFunctionName(int index) const {
    ASSERT_VALID_INDEX(index, functions);
    return functions[index].name;
}

int CustomResiduePairForce::addFunction(const std::string& name, const std::vector<double>& values, double min, double max) {
    functions.push_back(FunctionInfo(name, new Continuous1DFunction(values, min, max)));
    return functions.size()-1;
}

void CustomResiduePairForce::getFunctionParameters(int index, std::string& name, std::vector<double>& values, double& min, double& max) const {
    ASSERT_VALID_INDEX(index, functions);
    Continuous1DFunction* function = dynamic_cast<Continuous1DFunction*>(functions[index].function);
    if (function == NULL)
        throw OpenMMException("CustomResiduePairForce: function is not a Continuous1DFunction");
    name = functions[index].name;
    function->getFunctionParameters(values, min, max);
}

void CustomResiduePairForce::setFunctionParameters(int index, const std::string& name, const std::vector<double>& values, double min, double max) {
    ASSERT_VALID_INDEX(index, functions);
    Continuous1DFunction* function = dynamic_cast<Continuous1DFunction*>(functions[index].function);
    if (function == NULL)
        throw OpenMMException("CustomResiduePairForce: function is not a Continuous1DFunction");
    functions[index].name = name;
    function->setFunctionParameters(values, min, max);
}

ForceImpl* CustomResiduePairForce::createImpl() const {
    return new CustomResiduePairForceImpl(*this);
}

void CustomResiduePairForce::updateParametersInContext(Context& context) {
    dynamic_cast<CustomResiduePairForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
