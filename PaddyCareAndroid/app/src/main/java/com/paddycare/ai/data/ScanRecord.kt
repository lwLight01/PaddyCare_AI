package com.paddycare.ai.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity for scan history records.
 */
@Entity(tableName = "scan_history")
data class ScanRecord(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val timestamp: Long = System.currentTimeMillis(),
    val imagePath: String,
    val status: String,           // "ok", "not_paddy", "low_confidence", "error"
    val diseaseName: String?,     // null if not_paddy or error
    val diseaseNameBn: String?,
    val confidence: Float?,
    val treatment: String?,
)
