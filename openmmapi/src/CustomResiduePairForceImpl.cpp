/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
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

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/CustomResiduePairForceImpl.h"
#include "openmm/kernels.h"
#include "lepton/Operation.h"
#include "lepton/Parser.h"
#include <sstream>

using namespace OpenMM;
using Lepton::CustomFunction;
using Lepton::ExpressionTreeNode;
using Lepton::Operation;
using Lepton::ParsedExpression;
using std::map;
using std::pair;
using std::vector;
using std::set;
using std::string;
using std::stringstream;

/**
 * This class serves as a placeholder for angles and dihedrals in expressions.
 */
class CustomResiduePairForceImpl::FunctionPlaceholder : public CustomFunction {
public:
    int numArguments;
    FunctionPlaceholder(int numArguments) : numArguments(numArguments) {
    }
    int getNumArguments() const {
        return numArguments;
    }
    double evaluate(const double* arguments) const {
        return 0.0;
    }
    double evaluateDerivative(const double* arguments, const int* derivOrder) const {
        return 0.0;
    }
    CustomFunction* clone() const {
        return new FunctionPlaceholder(numArguments);
    }
};

CustomResiduePairForceImpl::CustomResiduePairForceImpl(const CustomResiduePairForce& owner) : owner(owner) {
}

CustomResiduePairForceImpl::~CustomResiduePairForceImpl() {
}

void CustomResiduePairForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcCustomResiduePairForceKernel::Name(), context);

    // Check for errors in the specification of parameters and exclusions.

    const System& system = context.getSystem();
    vector<set<int> > exclusions(owner.getNumDonors());
    vector<double> parameters;
    int numDonorParameters = owner.getNumPerDonorParameters();
    for (int i = 0; i < owner.getNumDonors(); i++) {
        int d1, d2, d3, d4;
        owner.getDonorParameters(i, d1, d2, d3, d4, parameters);
        if (d1 < 0 || d1 >= system.getNumParticles()) {
            stringstream msg;
            msg << "CustomResiduePairForce: Illegal particle index for a donor: ";
            msg << d1;
            throw OpenMMException(msg.str());
        }
        if (d2 < -1 || d2 >= system.getNumParticles()) {
            stringstream msg;
            msg << "CustomResiduePairForce: Illegal particle index for a donor: ";
            msg << d2;
            throw OpenMMException(msg.str());
        }
        if (d3 < -1 || d3 >= system.getNumParticles()) {
            stringstream msg;
            msg << "CustomResiduePairForce: Illegal particle index for a donor: ";
            msg << d3;
            throw OpenMMException(msg.str());
        }
        if (d4 < -1 || d4 >= system.getNumParticles()) {
          stringstream msg;
          msg << "CustomResiduePairForce: Illegal particle index for a donor: ";
          msg << d4;
          throw OpenMMException(msg.str());
        }
        if (parameters.size() != numDonorParameters) {
            stringstream msg;
            msg << "CustomResiduePairForce: Wrong number of parameters for donor ";
            msg << i;
            throw OpenMMException(msg.str());
        }
    }
    int numAcceptorParameters = owner.getNumPerAcceptorParameters();
    for (int i = 0; i < owner.getNumAcceptors(); i++) {
        int a1, a2, a3, a4;
        owner.getAcceptorParameters(i, a1, a2, a3, a4,parameters);
        if (a1 < 0 || a1 >= system.getNumParticles()) {
            stringstream msg;
            msg << "CustomResiduePairForce: Illegal particle index for an acceptor: ";
            msg << a1;
            throw OpenMMException(msg.str());
        }
        if (a2 < -1 || a2 >= system.getNumParticles()) {
            stringstream msg;
            msg << "CustomResiduePairForce: Illegal particle index for an acceptor: ";
            msg << a2;
            throw OpenMMException(msg.str());
        }
        if (a3 < -1 || a3 >= system.getNumParticles()) {
          stringstream msg;
          msg << "CustomResiduePairForce: Illegal particle index for an acceptor: ";
          msg << a3;
          throw OpenMMException(msg.str());
        }
        if (a4 < -1 || a4 >= system.getNumParticles()) {
          stringstream msg;
          msg << "CustomResiduePairForce: Illegal particle index for an acceptor: ";
          msg << a4;
          throw OpenMMException(msg.str());
        }
        if (parameters.size() != numAcceptorParameters) {
            stringstream msg;
            msg << "CustomResiduePairForce: Wrong number of parameters for acceptor ";
            msg << i;
            throw OpenMMException(msg.str());
        }
    }
    for (int i = 0; i < owner.getNumExclusions(); i++) {
        int donor, acceptor;
        owner.getExclusionParticles(i, donor, acceptor);
        if (donor < 0 || donor >= owner.getNumDonors()) {
            stringstream msg;
            msg << "CustomResiduePairForce: Illegal donor index for an exclusion: ";
            msg << donor;
            throw OpenMMException(msg.str());
        }
        if (acceptor < 0 || acceptor >= owner.getNumAcceptors()) {
            stringstream msg;
            msg << "CustomResiduePairForce: Illegal acceptor index for an exclusion: ";
            msg << acceptor;
            throw OpenMMException(msg.str());
        }
        if (exclusions[donor].count(acceptor) > 0) {
            stringstream msg;
            msg << "CustomResiduePairForce: Multiple exclusions are specified for donor ";
            msg << donor;
            msg << " and acceptor ";
            msg << acceptor;
            throw OpenMMException(msg.str());
        }
        exclusions[donor].insert(acceptor);
    }
    if (owner.getNonbondedMethod() == CustomResiduePairForce::CutoffPeriodic) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("CustomResiduePairForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
    kernel.getAs<CalcCustomResiduePairForceKernel>().initialize(context.getSystem(), owner);
}

double CustomResiduePairForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcCustomResiduePairForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> CustomResiduePairForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcCustomResiduePairForceKernel::Name());
    return names;
}

map<string, double> CustomResiduePairForceImpl::getDefaultParameters() {
    map<string, double> parameters;
    for (int i = 0; i < owner.getNumGlobalParameters(); i++)
        parameters[owner.getGlobalParameterName(i)] = owner.getGlobalParameterDefaultValue(i);
    return parameters;
}

ParsedExpression CustomResiduePairForceImpl::prepareExpression(const CustomResiduePairForce& force, const map<string, CustomFunction*>& customFunctions, map<string, vector<int> >& distances,
        map<string, vector<int> >& angles,map<string, vector<int> >& vectorangles, map<string, vector<int> >& dihedrals) {
    CustomResiduePairForceImpl::FunctionPlaceholder custom(1);
    CustomResiduePairForceImpl::FunctionPlaceholder distance(2);
    CustomResiduePairForceImpl::FunctionPlaceholder angle(3);
    CustomResiduePairForceImpl::FunctionPlaceholder vectorangle(4);
    CustomResiduePairForceImpl::FunctionPlaceholder dihedral(4);
    map<string, CustomFunction*> functions = customFunctions;
    functions["distance"] = &distance;
    functions["angle"] = &angle;
    functions["vectorangle"] = &vectorangle;
    functions["dihedral"] = &dihedral;
    ParsedExpression expression = Lepton::Parser::parse(force.getEnergyFunction(), functions);
    map<string, int> atoms;
    atoms["a1"] = 0;
    atoms["a2"] = 1;
    atoms["a3"] = 2;
    atoms["a4"] = 3;
    atoms["d1"] = 4;
    atoms["d2"] = 5;
    atoms["d3"] = 6;
    atoms["d4"] = 7;
    set<string> variables;
    for (int i = 0; i < force.getNumPerDonorParameters(); i++)
        variables.insert(force.getPerDonorParameterName(i));
    for (int i = 0; i < force.getNumPerAcceptorParameters(); i++)
        variables.insert(force.getPerAcceptorParameterName(i));
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        variables.insert(force.getGlobalParameterName(i));
    return ParsedExpression(replaceFunctions(expression.getRootNode(), atoms, distances, angles, vectorangles, dihedrals, variables)).optimize();
}

ExpressionTreeNode CustomResiduePairForceImpl::replaceFunctions(const ExpressionTreeNode& node, map<string, int> atoms,
        map<string, vector<int> >& distances, map<string, vector<int> >& angles,map<string, vector<int> >& vectorangles ,map<string, vector<int> >& dihedrals, set<string>& variables) {
    const Operation& op = node.getOperation();
    if (op.getId() == Operation::VARIABLE && variables.find(op.getName()) == variables.end())
        throw OpenMMException("CustomResiduePairForce: Unknown variable '"+op.getName()+"'");
    if (op.getId() != Operation::CUSTOM || (op.getName() != "distance" && op.getName() != "angle" && op.getName() != "vectorangle" && op.getName() != "dihedral"))
    {
        // This is not a distance, angle, vectorangle, or dihedral, so process its children.

        vector<ExpressionTreeNode> children;
        for (auto& child : node.getChildren())
            children.push_back(replaceFunctions(child, atoms, distances, angles, vectorangles, dihedrals, variables));
        return ExpressionTreeNode(op.clone(), children);
    }
    const Operation::Custom& custom = static_cast<const Operation::Custom&>(op);

    // Identify the atoms this term is based on.

    int numArgs = custom.getNumArguments();
    vector<int> indices(numArgs);
    for (int i = 0; i < numArgs; i++) {
        map<string, int>::const_iterator iter = atoms.find(node.getChildren()[i].getOperation().getName());
        if (iter == atoms.end())
            throw OpenMMException("CustomResiduePairForce: Unknown particle '"+node.getChildren()[i].getOperation().getName()+"'");
        indices[i] = iter->second;
    }
    
    // Select a name for the variable and add it to the appropriate map.
    
    stringstream variable;
    if (numArgs == 2)
        variable << "distance";
    else if (numArgs == 3)
        variable << "angle";
    else if (op.getName() == "vectorangle")
        variable << "vectorangle";
    else
        variable << "dihedral";
    for (int i = 0; i < numArgs; i++)
        variable << indices[i];
    string name = variable.str();
    if (numArgs == 2)
        distances[name] = indices;
    else if (numArgs == 3)
        angles[name] = indices;
    else if (op.getName() == "vectorangle")
        vectorangles[name] = indices;
    else
        dihedrals[name] = indices;
    
    // Return a new node that represents it as a simple variable.
    
    return ExpressionTreeNode(new Operation::Variable(name));
}

void CustomResiduePairForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcCustomResiduePairForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}
