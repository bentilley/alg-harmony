
#pragma once

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <string>
#include <stdexcept>
#include <map>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <tensorflow/core/util/events_writer.h>
#include "../JuceLibraryCode/JuceHeader.h"

#include "../engine/Plane.h"
#include "../util/Utilities.h"

//#include <Python.h>

using namespace std;
using namespace tensorflow;

namespace harmony {

    typedef std::pair<string, tensorflow::Tensor> TensorflowArg;
    typedef std::vector<TensorflowArg> TensorflowArgs;
    typedef std::map<string, tensorflow::Tensor*> TensorflowArgsMap;
    typedef std::vector<tensorflow::Tensor> TensorflowReturns;

    //int main(int argc, char *argv[])
    class Tensorflow : public AudioProcessorValueTreeState::Listener,
                       public ActionListener
    {
    public:

        Tensorflow(AudioProcessorValueTreeState& parameters, Plane& plane)
                : isReady(false),
                  parameters(parameters),
                  plane(plane) {

            //Init();
            //processBlock();
        }

        ~Tensorflow() {

            // Free any resources used by the session
            if (session != nullptr)
                session->Close();
        }

        void Init() {

            arguments = new TensorflowArgs{
                {"audio_inputs", {tensorflow::DT_FLOAT, tensorflow::TensorShape({120})}},
                {"midi_inputs", {tensorflow::DT_FLOAT, tensorflow::TensorShape({120})}},
                {"audio_input_gain", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"midi_input_gain", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"potential_decay", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"play_decay", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"play_replenishment", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"energy_decay", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"brightness", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"inhibition_rate", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"inhibition_gain", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"play_speed", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"play_spacing", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"play_appetite", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"play_vigor", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"resonate_gain", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}},
                {"reverse_resonate_gain", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}}
                //{"potential_grouping", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}}
                //{"synth/inputs", {tensorflow::DT_FLOAT, tensorflow::TensorShape({1})}}
                //{"synth/num_samples", {tensorflow::DT_INT32, tensorflow::TensorShape({1})}}
            };

            //parameters.addParameterListener("potential_decay", this);

            std::for_each(arguments->begin(),
                          arguments->end(),
                          [&](std::pair<std::string, tensorflow::Tensor>& argument){
                              //if (argument.first != "midi_inputs") {
                              float* value = parameters.getRawParameterValue(argument.first.c_str());
                              if (value != nullptr) {
                                  parameters.addParameterListener(argument.first.c_str(), this);
                                  argument.second.scalar<float>()() = *value;
                              }
                              //}
                          });

            returns = new TensorflowReturns();

            //load();

            MessageManager* messageManager =  MessageManager::getInstance();
            messageManager->registerBroadcastListener(this);
        }

        void actionListenerCallback(const String& message) {

            if (message == "load_graph") {
                load();
            }
        }

        void parameterChanged(const String&	parameterId,
                              float newValue) override {

            printf("parameterChanged id=%s value=%f\n",
                   static_cast<const char*>(parameterId.toUTF8()),
                   newValue);

            auto argument = findArgument(parameterId);
            /*
            auto argument = std::find_if(arguments->begin(),
                                         arguments->end(),
                                         [&](const std::pair<std::string, tensorflow::Tensor>& element){
                                             return element.first == static_cast<const char*>(parameterId.toUTF8());
                                         });
            */

            argument.scalar<float>()() = newValue;

            //memcpy(argument->second.scalar<float>().data(), &newValue, 1);

            /*
            //arguments->at(0).second.flat<float>().data();
            std::copy_n(plane.currentTick().inputs.begin(),
                        plane.currentTick().inputs.size(),
                        arguments->at(0).second.flat<float>().data());
            */
        }

        tensorflow::Tensor& findArgument(const String& parameterId) {

            return std::find_if(arguments->begin(),
                                arguments->end(),
                                [&](const std::pair<std::string, tensorflow::Tensor>& element){
                                    return element.first == static_cast<const char*>(parameterId.toUTF8());
                                })->second;
            //return argument;
        }

        void load() {

            isReady = false;

            if (session != nullptr) {
                session->Close();
                session = nullptr;
            }

            // Initialize a tensorflow session
            Status status = NewSession(SessionOptions(), &session);
            if (!status.ok()) {
                //std::cout << status.ToString() << "\n";
                throw std::runtime_error( "Could not create Tensorflow session: " + status.ToString());
            }

            status = ReadTextProto(Env::Default(), "/opt/harmony-engine/graph.pb", &graphDef);
            //status = ReadTextProto(Env::Default(), "../../../../Source/tensorflow/models/graph.pb", &graphDef);
            Logger::writeToLog("status.ok() " + std::to_string(status.ok()));
            Logger::writeToLog("status.ToString() " + status.ToString());
            if (!status.ok())
                throw std::runtime_error("Could not read Tensorflow graph: " + status.ToString());

            status = ReadTextProto(Env::Default(), "/opt/harmony-engine/resonate.pb", &resonateDef);
            if (!status.ok())
                throw std::runtime_error("Could not read Resonate graph: " + status.ToString());

            /*
            status = ReadTextProto(Env::Default(), "/opt/harmony-engine/synth.pb", &synthDef);
            if (!status.ok())
                throw std::runtime_error("Could not read synth graph: " + status.ToString());
            */

            //graphDef.MergeFrom(resonateDef);

            // Add the graph to the session
            status = session->Create(graphDef);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                throw std::runtime_error( "Could not create Tensorflow graph: " + status.ToString());
            }

            /*
            status = session->Extend(resonateDef);
            if (!status.ok())
                throw std::runtime_error( "Could not merge resonate graph: " + status.ToString());

            status = session->Extend(synthDef);
            if (!status.ok())
                throw std::runtime_error( "Could not merge synth graph: " + status.ToString());
            */

            status = session->Run({}, {}, {"init_all_vars_op"}, nullptr);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                throw std::runtime_error( "Could not initialize variables in Tensorflow session: " + status.ToString());
            }

            /*
            status = session->Run({}, {}, {"synth/init_all_vars_op"}, nullptr);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                throw std::runtime_error( "Could not initialize synth variables in Tensorflow session: " + status.ToString());
            }
            */

            isReady = true;
        }

        void prepareToPlay(double sampleRate, int maximumBlockSize) {

            /*
            auto argument = findArgument("num_samples");
            argument.scalar<int>()() = maximumBlockSize;
            */
        }

        void processBlock(AudioSampleBuffer& buffer) { //std::vector<float>& midiInputs, std::vector<float>& outputs) { //AudioSampleBuffer& buffer, MidiBuffer& midiMessages) {
            
            if (!isReady)
                return;
            
            //Logger::writeToLog("Tensorflow processBlock");

            using namespace std::chrono;
            //tick_ptr tick = Tick::create();

            tick_ptr tick;
            if (!plane.back(tick))
                return;

            std::copy_n(tick->audioInputs.begin(),
                        tick->audioInputs.size(),
                        arguments->at(0).second.flat<float>().data());

            std::copy_n(tick->midiInputs.begin(),
                        tick->midiInputs.size(),
                        arguments->at(1).second.flat<float>().data());

            //arguments->at(1).second.data() = buffer.getNumSamples();

            //high_resolution_clock::time_point t1 = high_resolution_clock::now();

            // Run the session, evaluating our "c" operation from the graph
            Status status = session->Run(*arguments,
                    {"potentials",
                     "inhibitions",
                     "players",
                     "energies",
                     "merged_summary"}, {}, &(*returns));
            //"synth/outputs"}, {}, &(*returns));
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                throw std::runtime_error( "Failed to run Tensorflow graph" );
            }
            
            // Trying to get the summary information out of the graph
            /***************
            ScopedPointer<TensorflowArgs> blank_args = new TensorflowArgs{};
            status = session->Run(*blank_args, {"merged_summary"}, {}, &(*summaries));

            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
                throw std::runtime_error( "Failed to run summary operation" );
            }
            ****************/
            
            //high_resolution_clock::time_point t2 = high_resolution_clock::now();
            //duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
            //Logger::writeToLog("Tensorflow processBlock took time " + std::to_string(time_span.count()));

            copyOutputs(tick, buffer);
            
            tick_num++;

            //auto output_c = outputs[0].scalar<float>();
            //std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
            //std::cout << (int)output_c() << "\n"; // 30
        }

        void copyOutputs(tick_ptr tick, AudioSampleBuffer& buffer) {

            //Tick* tick;
            //plane.prepareToWrite(tick);

            auto potentials = returns->at(0).vec<float>();
            std::copy(potentials.data(),
                      potentials.data() + potentials.size(),
                      tick->potentials.begin());

            auto inhibitions = returns->at(1).vec<float>();
            std::copy(inhibitions.data(),
                      inhibitions.data() + inhibitions.size(),
                      tick->inhibitions.begin());

            auto players = returns->at(2).vec<float>();
            std::copy(players.data(),
                      players.data() + players.size(),
                      tick->players.begin());
            //printf("tick->players.at(50)=%f\n", tick->players.at(50));

            auto energies = returns->at(3).vec<float>();
            std::copy(energies.data(),
                      energies.data() + energies.size(),
                      tick->energies.begin());
            //printf("tick->energies.at(50)=%f\n", tick->energies.at(50));
            
            
            std::string eventFile = "/opt/harmony-engine/logs/fromcpp/events";
            tensorflow::EventsWriter writer(eventFile);
            tensorflow::Event event;
            event.set_wall_time(tick_num * 20);
            event.set_step(tick_num);
            
            //Logger::writeToLog(returns->at(4).DebugString());
            auto merged_summary = returns->at(4).scalar<string>();
            string* summary_string = merged_summary.data();
            std::cout << "tensor content: " << *summary_string << std::endl;
            //tensorflow::Tensor summary(returns->at(4));
            //TensorProto protoSummary;
            //summary.AsProtoField(& protoSummary);
            //protoSummary.CopyFrom(summary);
            //summary.AsProtoTensorContent(& protoSummary);
            
            tensorflow::Summary::Value* summ_val = event.mutable_summary()->add_value();
            summ_val->set_tag("loss");
            summ_val->set_simple_value(150.f);
            std::string serialised_summary;
            event.summary().SerializeToString(&serialised_summary);
            
            std::cout << serialised_summary.length() << std::endl;
            //std::cout << protoSummary.tensor_content().length() << std::endl;
            
            //tensorflow::Summary new_summary;
            //bool parse_summary = new_summary.ParseFromString(serialised_summary);
            //printf("summary success?: %d\n", parse_summary);
            //new_summary.value()
            //Logger::writeToLog(protoSummary.DebugString());
            //Logger::writeToLog(protoSummary.Debug());
            //printf("isInitialised: %d\n", protoSummary.IsInitialized());
            
            bool status = event.mutable_summary()->ParsePartialFromString(*summary_string);
            printf("summary success?: %d\n", status);
            
            
            //Logger::writeToLog(event.summary().value_size());
            printf("num summaries: %d\n", event.summary().value_size());
            writer.WriteEvent(event);
            //HistogramProto* histogram;
            //histogram->
            //summ_val->set_allocated_histo(&(returns->at(4)));
            //event.mutable_summary()->
            
            //auto events = returns->at(4).vec<int>();
            
            
            /*
            auto copied_energies = tick->energies;
            std::cout << "energies: ";
            for (int i = 0; i < copied_energies.size(); i++)
                std::cout << copied_energies.at(i) << " ";
            std::cout << std::endl;
            */
             
            /*
            auto audio = returns->at(4).vec<float>();
            int numSamples = buffer.getNumSamples();
            std::copy(audio.data(),
                      audio.data() + numSamples,
                      buffer.getWritePointer(0));
            std::copy(audio.data(),
                    audio.data() + numSamples,
                    buffer.getWritePointer(1));
            */
            //tick->audio.begin());
            //printf("outputs.at(50)=%f\n", *(outputs.data()));
        }

        std::atomic<bool> isReady;

        Plane& plane;
        
        int tick_num = 1;

        AudioProcessorValueTreeState& parameters;

        Session* session = nullptr;

        GraphDef graphDef;

        GraphDef resonateDef;

        GraphDef synthDef;

        ScopedPointer<tensorflow::Tensor> midiInputs;

        ScopedPointer<TensorflowArgs> arguments;

        ScopedPointer<TensorflowReturns> returns;
        
        tensorflow::TensorProto* protoSummary = nullptr;
    };
}



        /*
        std::string fileName("harmony");
        std::string functionName("multiply");

        PyObject *pName, *pModule, *pDict, *pFunc;
        PyObject *pArgs, *pValue;
        int i;

        Py_Initialize();
        PyObject* sysPath = PySys_GetObject((char*)"path");
        PyObject* programName = PyUnicode_FromString("/Users/user/Bounce/harmony-engine/Source/tensorflow/");
        PyList_Append(sysPath, programName);

        pName = PyUnicode_DecodeFSDefault(fileName.c_str());
        // Error checking of pName left out

        pModule = PyImport_Import(pName);
        Py_DECREF(pName);

        if (pModule != NULL) {
            printf("Loaded a module");
            pFunc = PyObject_GetAttrString(pModule, functionName.c_str());
            // pFunc is a new reference

            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(2);
                for (i = 0; i < 2; ++i) {
                    pValue = PyLong_FromLong(atoi("3"));
                    //pValue = PyLong_FromLong(atoi(argv[i + 3]));
                    if (!pValue) {
                        Py_DECREF(pArgs);
                        Py_DECREF(pModule);
                        fprintf(stderr, "Cannot convert argument\n");
                        return 1;
                    }
                    // pValue reference stolen here:
                    PyTuple_SetItem(pArgs, i, pValue);
                }
                pValue = PyObject_CallObject(pFunc, pArgs);
                Py_DECREF(pArgs);
                if (pValue != NULL) {
                    printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                    Py_DECREF(pValue);
                }
                else {
                    Py_DECREF(pFunc);
                    Py_DECREF(pModule);
                    PyErr_Print();
                    fprintf(stderr,"Call failed\n");
                    return 1;
                }
            }
            else {
                if (PyErr_Occurred())
                    PyErr_Print();
                fprintf(stderr, "Cannot find function \"%s\"\n", functionName.c_str());
            }
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
        }
        else {
            PyErr_Print();
            fprintf(stderr, "Failed to load \"%s\"\n", fileName.c_str());
            return 1;
        }
        Py_Finalize();

        /*

        PyObject *pName, *pModule, *pDict, *pFunc;
        PyObject *pArgs, *pValue;
        int i;

        Py_SetProgramName((wchar_t*)L"test");

        Py_Initialize();

        //PySys_SetArgv(argc, (wchar_t**)argv);
        PyRun_SimpleString("import tensorflow as tf\n"
                "print(tf.__version__)\n");

        PyRun_SimpleString("import cv2\n"
                "print(cv2.__version__)\n");

        QString qs = QDir::currentPath();
        std::wstring ws = qs.toStdWString();
        PySys_SetPath(ws.data());
        pName = PyUnicode_DecodeFSDefault(fileName.c_str());
        //pName = PyUnicode_DecodeFSDefault(argv[1]);
        // Error checking of pName left out

        pModule = PyImport_Import(pName);
        Py_DECREF(pName);

        if (pModule != NULL) {
            pFunc = PyObject_GetAttrString(pModule, functionName.c_str());
            // pFunc is a new reference

            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(argc - 3);
                //for (i = 0; i < argc - 3; ++i) {
                //    pValue = PyLong_FromLong(atoi(argv[i + 3]));
                //    if (!pValue) {
                //        Py_DECREF(pArgs);
                //        Py_DECREF(pModule);
                //        fprintf(stderr, "Cannot convert argument\n");
                //        return 1;
                //    }
                //    /* pValue reference stolen here:
                //    PyTuple_SetItem(pArgs, i, pValue);
                //}
                pValue = PyObject_CallObject(pFunc, pArgs);
                Py_DECREF(pArgs);
                if (pValue != NULL) {
                    printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                    Py_DECREF(pValue);
                }
                else {
                    Py_DECREF(pFunc);
                    Py_DECREF(pModule);
                    PyErr_Print();
                    fprintf(stderr,"Call failed\n");
                    //return 1;
                }
            }
            else {
                if (PyErr_Occurred())
                    PyErr_Print();
                fprintf(stderr, "Cannot find function \"%s\"\n", functionName.c_str());
            }
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
        }
        else {
            PyErr_Print();
            fprintf(stderr, "Failed to load \"%s\"\n", fileName.c_str());
            //return 1;
        }
        Py_Finalize();
        */
