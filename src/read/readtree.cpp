#include "wet.h"
#include "forest.h"
#include "tool.h"
#include "read.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
using namespace rapidjson;

std::map<std::string,int> type_map = {{"CT_OPS",CT_OPS},
                                      {"CT_VAR",CT_VAR},
                                      {"CT_CON",CT_CON},
                                      {"CT_PLUS",CT_PLUS},
                                      {"CT_MUL2",CT_MUL2},
                                      {"CT_WEIGHTED_PLUS",CT_WEIGHTED_PLUS},
                                      {"CT_WEIGHTED_PLUS_2",CT_WEIGHTED_PLUS_2},
                                      {"CT_MINUS",CT_MINUS}};

std::map<std::string,int> operator_map = {{"CT_OPS",1},
                                      {"CT_VAR",0},
                                      {"CT_CON",0},
                                      {"CT_PLUS",1},
                                      {"CT_MUL2",1},
                                      {"CT_WEIGHTED_PLUS",1},
                                      {"CT_WEIGHTED_PLUS_2",1},
                                      {"CT_MINUS",1}};

Forest parse_forest(std::string filepath){
    printf("file name : %s\n",filepath.c_str());
    std::ifstream ifs(filepath);
    IStreamWrapper isw(ifs);
    Document d;
    d.ParseStream(isw);
    assert(d.IsObject());
    assert(d.HasMember("treename"));
    assert(d["treename"].IsString());
    printf("tree name is %s\n",d["treename"].GetString());

    Forest f;
    f.fdim = d["fdim"].GetUint();
    f.xdim = d["xdim"].GetUint();
    register_node(f.xdim);
    const Value& constant_range = d["constant_range"];
    std::pair<double,double> cr = {constant_range[0].GetDouble(),constant_range[1].GetDouble()};
    int randseed = d["randseed"].GetInt();
    std::map<unsigned,VEC_DOUBLE> initial_value;
    std::vector<unsigned> untrainable;
    const Value& forest = d["forest"];
    std::cout << forest.IsObject() << "," <<forest.IsArray() << std::endl;
    for (Value::ConstValueIterator itr = forest.Begin(); itr != forest.End(); ++itr){
        int idx = f.trees.size();
        unsigned root_id = (*itr)["tree"].GetUint();
        f.root_id.push_back(root_id);
        f.trees.push_back(Tree());
        f.trees[idx].node_names.clear();
        const Value& nodes = (*itr)["nodes"];
        for ( Value::ConstValueIterator node = nodes.Begin(); node != nodes.End();++node) {
            std::string node_type = (*node)["type"].GetString();
            if (node_type != "*") {
                int node_type_ = type_map[node_type];
                unsigned node_id = (*node)["node_id"].GetUint();
                f.trees[idx].node_names.push_back(node_id);
                f.trees[idx].node_type.insert(std::pair<unsigned, int>(node_id, node_type_));
                if (operator_map[node_type]) {
                    f.trees[idx].operator_nodes.push_back(node_id);
                } else {
                    f.trees[idx].terminal_nodes.push_back(node_id);
                }
                // connection_down
                VEC_ID c_down;
                for (auto &v: (*node)["children"].GetArray()) {
                    c_down.push_back(v.GetUint());
                }
                f.trees[idx].connection_down.insert(std::pair<unsigned, VEC_ID>(node_id, VEC_ID(c_down)));
                // weight and trainable
                if (!(*node)["trainable"].GetBool()) {
                    untrainable.push_back(node_id);
                }
                VEC_DOUBLE i_weight;

                for (auto &v: (*node)["weights"].GetArray()) {
                    i_weight.push_back(v.GetDouble());
                }
                if (!i_weight.empty())
                    initial_value.insert(std::pair<unsigned, VEC_DOUBLE>(node_id, VEC_DOUBLE(i_weight)));

            }else{
                int node_type_ = type_map["CT_OPS"];
                unsigned node_id = (*node)["node_id"].GetUint();
                int head_length = (*node)["children"].GetInt();

                std::vector<unsigned> idset;
                idset.push_back(node_id);
                for (int j =0;j< head_length + 5 * (head_length+1) - 1 ;j++){
                    idset.push_back(get_id());
                }

                for (unsigned int i = 0; i < head_length*2+1; i++)
                {
                    if (i < head_length)
                    {
                        f.trees[idx].node_names.push_back(idset[i]);
                        f.trees[idx].operator_nodes.push_back(idset[i]);
                        f.trees[idx].connection_down.insert(std::pair<unsigned, VEC_ID>(idset[i], {idset[2 * i + 1], idset[2 * i + 2]}));
                        f.trees[idx].node_type.insert(std::pair<unsigned, int>(idset[i], CT_OPS));
                    }
                    else
                    {
                        f.trees[idx].node_names.push_back(idset[i]);
                        f.trees[idx].operator_nodes.push_back(idset[i]);
                        f.trees[idx].connection_down.insert(std::pair<unsigned, VEC_ID>(idset[i], {idset[(i-head_length)*4+head_length*2+1],idset[(i-head_length)*4+1+head_length*2+1]}));
                        f.trees[idx].node_type.insert(std::pair<unsigned, int>(idset[i], CT_PLUS));

                        f.trees[idx].node_names.push_back(idset[(i-head_length)*4+head_length*2+1]);
                        f.trees[idx].terminal_nodes.push_back(idset[(i-head_length)*4+head_length*2+1]);
                        f.trees[idx].connection_down.insert(std::pair<unsigned, VEC_ID>(idset[(i-head_length)*4+head_length*2+1], {}));
                        f.trees[idx].node_type.insert(std::pair<unsigned, int>(idset[(i-head_length)*4+head_length*2+1], CT_CON));

                        f.trees[idx].node_names.push_back(idset[(i-head_length)*4+1+head_length*2+1]);
                        f.trees[idx].terminal_nodes.push_back(idset[(i-head_length)*4+1+head_length*2+1]);
                        f.trees[idx].connection_down.insert(std::pair<unsigned, VEC_ID>(idset[(i-head_length)*4+1+head_length*2+1], {idset[(i-head_length)*4+2+head_length*2+1], idset[(i-head_length)*4+3+head_length*2+1]}));
                        f.trees[idx].node_type.insert(std::pair<unsigned, int>(idset[(i-head_length)*4+1+head_length*2+1], CT_MUL2));

                        f.trees[idx].node_names.push_back(idset[(i-head_length)*4+2+head_length*2+1]);
                        f.trees[idx].terminal_nodes.push_back(idset[(i-head_length)*4+2+head_length*2+1]);
                        f.trees[idx].connection_down.insert(std::pair<unsigned, VEC_ID>(idset[(i-head_length)*4+2+head_length*2+1], {}));
                        f.trees[idx].node_type.insert(std::pair<unsigned, int>(idset[(i-head_length)*4+2+head_length*2+1], CT_CON));

                        f.trees[idx].node_names.push_back(idset[(i-head_length)*4+3+head_length*2+1]);
                        f.trees[idx].terminal_nodes.push_back(idset[(i-head_length)*4+3+head_length*2+1]);
                        f.trees[idx].connection_down.insert(std::pair<unsigned, VEC_ID>(idset[(i-head_length)*4+3+head_length*2+1], {}));
                        f.trees[idx].node_type.insert(std::pair<unsigned, int>(idset[(i-head_length)*4+3+head_length*2+1], CT_VAR));
                    }
                }
            }
        }
        // gradient chain
        for (unsigned i = 0; i < f.trees[idx].node_names.size(); i++)
        {
            std::vector<unsigned> l;
            l.push_back(f.trees[idx].node_names[i]);
            f.trees[idx].connection_gradient.insert(std::pair<unsigned, VEC_ID>(f.trees[idx].node_names[i], {ID_N}));
            while (!l.empty())
            {
                auto c = l.back();
                l.pop_back();
                f.trees[idx].connection_gradient[f.trees[idx].node_names[i]].push_back(c);
                std::for_each(f.trees[idx].connection_down[c].begin(), f.trees[idx].connection_down[c].end(), [&](unsigned n)
                { l.push_back(n); });
            }
        }
        // layer
        f.trees[idx].layer_nodes.push_back({root_id});
        std::vector<unsigned> l = f.trees[idx].layer_nodes[0];

        while (!l.empty())
        {
            std::vector<unsigned> l2;
            for (unsigned i = 0; i < l.size(); i++)
            {
                for (unsigned j = 0; j < f.trees[idx].connection_down[l[i]].size(); j++)
                {
                    l2.push_back(f.trees[idx].connection_down[l[i]][j]);
                }
            }
            if (l2.size() != 0)
            {
                f.trees[idx].layer_nodes.push_back({});
                f.trees[idx].layer_nodes.back() = l2;
            }
            l = l2;
        }
    }// build tree

    std::vector<Weights> trees_weights;
    std::vector<Progress> trees_progress;
    for (unsigned i = 0; i < f.fdim; i++)
    {
        trees_weights.push_back(generate_weights_from_tree(f.trees[i], f.xdim, 1, cr, randseed));
        trees_progress.push_back(generate_progress_from_tree(f.trees[i]));
    }
    f.trees_weights = trees_weights[0];
    f.trees_progress = trees_progress[0];
    for (unsigned i = 1; i < f.fdim; i++)
    {
        f.trees_weights = merge_weights(f.trees_weights, trees_weights[i]);
        f.trees_progress = merge_progress(f.trees_progress, trees_progress[i]);
    }
    for (auto it : initial_value){
        f.trees_weights.weights[it.first] = VEC_DOUBLE(initial_value[it.first]);
    }
    f.trees_weights.untrainable=VEC_ID(untrainable);
    return f;
}