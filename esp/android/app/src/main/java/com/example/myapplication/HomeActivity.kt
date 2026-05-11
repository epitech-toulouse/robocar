package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.material3.Surface
import com.example.myapplication.navigation.AppRoot
import com.example.myapplication.ui.theme.MyApplicationTheme

class HomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MyApplicationTheme {
                Surface {
                    AppRoot(
                        onOpenOpenCvCamera = {
                            startActivity(Intent(this, MainActivity::class.java))
                        },
                        onOpenCommandCenter = {
                            startActivity(Intent(this, CommandActivity::class.java))
                        }
                    )
                }
            }
        }
    }
}
