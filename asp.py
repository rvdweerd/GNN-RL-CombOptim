from graph import Graph
import clingo

def Solve_MinVertexCover_ASP(G):
    asp_program = "#const n={}.\n".format(G.num_nodes)
    for n,elist in G.out_edges.items():
            for e in elist:
                asp_program += "edge("+str(n)+","+str(e[0])+").\n"
    asp_program += """
        edge(A,B):-edge(B,A).
        node(0..n-1).
        {choose(N) : node(N)}.

        edge_connected(A,B) :- choose(A), node(B).
        edge_connected(A,B):- choose(B), node(A).
        :- edge(A,B), not edge_connected(A,B).

        count(N) :- N=#count{P : choose(P)}.
        #minimize{N:count(N)}.

        #show choose/1.
        %#show seen/1.
    """
    # control = clingo.Control()
    # control.add("base", [], asp_program)
    # control.ground([("base", [])])
    # control.configuration.solve.models = 0
    # solutions=[]
    # with control.solve(yield_ = True) as handle:
    #     for model in handle:
    #         solution = []
    #         for atom in model.symbols(shown = True):
    #             if atom.name == "choose":
    #                 solution.append(atom.arguments[0].number)
    #         print("Solution: \n")
    #         print(solution)
    #         solutions.append(solution)
    #         #yield(solution)
    # return solutions
    control = clingo.Control()
    control.add("base", [], asp_program)
    control.ground([("base", [])])
    # Define a function that will be called when an answer set is found
    # This function sorts the answer set alphabetically, and prints it
    solutions=[]
    def on_model(model):
        if model.optimality_proven == True:
            #sorted_model = [str(atom) for atom in model.symbols(shown=True)]
            sorted_model = [atom.arguments[0].number if atom.name == "choose" else "" for atom in model.symbols(shown=True)]
            sorted_model.sort()
            solutions.append(sorted_model)
            #print("Optimal answer set: {{{}}}".format(", ".join(sorted_model)))
    # Ask clingo to find all optimal models (using an upper bound of 0 gives all models)
    control.configuration.solve.opt_mode = "optN"
    control.configuration.solve.models = 0
    # Call the clingo solver, passing on the function on_model for when an answer set is found
    answer = control.solve(on_model=on_model)
    # Print a message when no answer set was found
    #if answer.satisfiable == False:
    #    print("No answer sets")
    return solutions
