import mongoose, { Schema, Document } from 'mongoose';

export interface IUserDocument extends Document {
  email: string;
  name: string;
  password?: string;
  avatar?: string;
  authProvider: 'local' | 'google';
  googleId?: string;
  consultations: mongoose.Types.ObjectId[];
  createdAt: Date;
  updatedAt: Date;
}

const UserSchema = new Schema<IUserDocument>(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
     
      index: true,
    },
    name: {
      type: String,
      required: true,
      trim: true,
    },

    password:{
      type: String,
    },

    avatar:{
      type:String,
      default: '',
    },


    authProvider:{
      type: String,
      enum: ['local', 'google'],
      required: true,
      default: 'local',
    },

    
    googleId: {
      type: String,
      sparse: true,
      index: true,
    },
    consultations: [{
      type: Schema.Types.ObjectId,
      ref: 'Consultation',
    }],
  },
  {
    timestamps: true,
  
    toJSON: { virtuals: true },
    toObject: { virtuals: true },
  }
);


UserSchema.index({ email: 1, createdAt: -1 });

export const User = mongoose.model<IUserDocument>('User', UserSchema);