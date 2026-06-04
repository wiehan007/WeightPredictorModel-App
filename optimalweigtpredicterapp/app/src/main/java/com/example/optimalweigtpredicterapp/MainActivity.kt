package com.example.optimalweigtpredicterapp

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

class MainActivity : ComponentActivity() {
    private var helper: TFLiteHelper? = null
    private val scope = MainScope()

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        // show UI immediately
        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) {
                    PredictorScreen(
                        onPredict = { h -> helper?.predictWeightKg(h) ?: Float.NaN }
                    )
                }
            }
        }

        // load the model
        scope.launch(Dispatchers.Default) {
            val result = runCatching {
                TFLiteHelper(this@MainActivity)
            }
            withContext(Dispatchers.Main) {
                result.onSuccess { h ->
                    helper = h
                    Toast.makeText(this@MainActivity, "Model is ready", Toast.LENGTH_SHORT)
                        .show()
                }
                .onFailure {
                    Log.e("TFLite", "Model load failed", it)
                    Toast.makeText(this@MainActivity, "Model failed to load", Toast.LENGTH_LONG)
                        .show()
                }
            }
        }
    }

    override fun onDestroy(){
        super.onDestroy()
        runCatching {helper?.close()}
        scope.cancel()
    }
}

class TFLiteHelper(context :Context){
    private val interpreter : Interpreter
    init{
        val assetName = "height_weight.tflite"
        val bytes = context.assets.open(assetName).use{
            it.readBytes()
        }
        Log.d("TFLite", "asset =$assetName size=${bytes.size} shal6=${shal6(bytes)}")

        val bb = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        bb.put(bytes).rewind()

        val opts = Interpreter.Options().apply{
            setUseXNNPACK(false)
            setNumThreads(1)
        }

        interpreter = Interpreter(bb,opts)

        fun info(t: org.tensorflow.lite.Tensor) = "shape${t.shape().contentToString()} type=${t.dataType()}"

        Log.d("TFLITE","INPUT -> ${info(interpreter.getInputTensor(0))}")
        Log.d("TFLITE","OUTPUT -> ${info(interpreter.getOutputTensor(0))}")
    }

    private fun shal6 (b: ByteArray):String {
        val md = MessageDigest.getInstance("SHA-256").digest(b)
        return md.take(8).joinToString(""){
            "%02x".format(it)
        }
    }

    private fun safePredict (h: Float): Float {
        val int = interpreter.getInputTensor(0)
        val out = interpreter.getOutputTensor(0)

        require(int.dataType().name == "FLOAT32" && out.dataType().name == "FLOAT32") {
            "Expected Float32 model got in=${int.dataType()} out=${out.dataType()}"
        }

        val inputObj: Any = when (int.shape().size) {
            1 -> floatArrayOf(h) //[1]
            2 -> arrayOf(floatArrayOf(h)) //[1,1]
            else -> error("Unsupported input shape ${int.shape().contentToString()}")
        }

        val outputObj: Any = when (out.shape().size) {
            1 -> FloatArray(1)
            2 -> arrayOf(FloatArray(1))
            else -> error("Unsupported output shape ${out.shape().contentToString()}")
        }

        interpreter.run(inputObj, outputObj)

        val y = when (outputObj) {
            is FloatArray -> outputObj[0]
            is Array<*> -> (outputObj[0] as FloatArray)[0]
            else -> Float.NaN
        }

        return if (y.isFinite()) y else Float.NaN
    }
    fun predictWeightKg(heightCm:Float) : Float = try {
        val y = safePredict(heightCm)
        Log.d("TFLite","predict h=$heightCm -> $y")
        y
    } catch (t: Throwable){
        Log.e("TFLite", "predict failed", t)
        Float.NaN
    }
    fun close() = interpreter.close()
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PredictorScreen(onPredict:(Float) -> Float) {
    var heightText by remember { mutableStateOf("")}
    var result by remember { mutableStateOf<Float?>(null)}

    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ){
        Text("Height -> Height (TFLite demo)", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(16.dp))

        OutlinedTextField(
            value = heightText,
            onValueChange = { heightText = it},
            label = { Text ("Height (cm)")},
            keyboardOptions = KeyboardOptions (keyboardType = KeyboardType.Number),
            singleLine = true
        )

        Spacer(Modifier.height(16.dp))

        Button(
            onClick = {
                val h = heightText.toFloatOrNull()
                result = h?.let(onPredict)
            }) {
            Text("predict")
        }

        Spacer(Modifier.height(24.dp))

        Text(
            text = when (val r = result){
                null -> "Enter a height and press Predict."
                Float.NaN -> "Prediction failed. See logcat: TFLite"
                else -> "Predicted weight: ${"%.1f".format(r)} kg"
            }
        )
    }
}

