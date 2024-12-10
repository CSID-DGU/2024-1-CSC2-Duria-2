package com.example.duria.services

import com.google.firebase.database.*
import com.example.duria.models.Message
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.google.firebase.database.FirebaseDatabase

object FirebaseService {
    private val database: DatabaseReference = FirebaseDatabase.getInstance().getReference("messages")
    private const val TAG = "FirebaseService"

    /**
     * 메시지를 Firebase Realtime Database에 저장하는 함수
     */
    fun sendMessage(message: Message) {
        val newMessageRef = database.push()

        // slots 키 정제
        val sanitizedSlots = sanitizeSlots(message.slots)
        // temp_slots 키 정제
        val sanitizedTempSlots = sanitizeSlots(message.temp_slots)

        // 로그로 sanitizedSlots와 sanitizedTempSlots 확인
        Log.d(TAG, "Sanitized Slots: $sanitizedSlots")
        Log.d(TAG, "Sanitized Temp_Slots: $sanitizedTempSlots")

        // 정제된 slots와 temp_slots를 사용하여 새로운 Message 객체 생성
        val sanitizedMessage = message.copy(
            slots = sanitizedSlots,
            temp_slots = sanitizedTempSlots
        )

        // Firebase에 정제된 메시지 저장
        newMessageRef.setValue(sanitizedMessage)
            .addOnSuccessListener {
                Log.d(TAG, "메시지 전송 성공")
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "메시지 전송 실패: ${e.message}")
            }
    }

    /**
     * Firebase Realtime Database에서 실시간으로 메시지를 수신하는 함수
     */
    fun observeMessages(onMessageAdded: (Message) -> Unit) {
        val childEventListener = object : ChildEventListener {
            override fun onChildAdded(snapshot: DataSnapshot, previousChildName: String?) {
                val message = snapshot.getValue(Message::class.java)
                if (message != null) {
                    // 메인 스레드에서 콜백 호출
                    Handler(Looper.getMainLooper()).post {
                        onMessageAdded(message)
                    }
                }
            }

            override fun onChildChanged(snapshot: DataSnapshot, previousChildName: String?) {}
            override fun onChildRemoved(snapshot: DataSnapshot) {}
            override fun onChildMoved(snapshot: DataSnapshot, previousChildName: String?) {}
            override fun onCancelled(error: DatabaseError) {
                println("Failed to read messages: ${error.toException()}")
            }
        }
        database.addChildEventListener(childEventListener)
    }

    /**
     * slots 및 temp_slots 맵의 키를 정제하는 함수
     */
    private fun sanitizeSlots(slots: Map<String, String>?): Map<String, String>? {
        return slots?.mapKeys { entry ->
            sanitizeKey(entry.key)
        }
    }

    /**
     * 개별 키를 정제하는 함수
     * 유효하지 않은 문자를 '_'로 대체
     */
    private fun sanitizeKey(key: String): String {
        val invalidChars = listOf("/", ".", "#", "$", "[", "]")
        var sanitizedKey = key
        invalidChars.forEach { char ->
            sanitizedKey = sanitizedKey.replace(char, "_")
        }
        return sanitizedKey
    }
}
