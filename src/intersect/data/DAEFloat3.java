/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package intersect.data;

/**
 * Wrapper around a float3 (x,y,z)
 * @author Koen Samyn
 */
public class DAEFloat3 {

    /**
     * Creates a new DAEFloat3 object, with the
     * members initialized to 0.
     */
    DAEFloat3(){
        x=0;
        y=0;
        z=0;
    }

    /**
     * Creates a new DAEFloat3 object.
     * @param xValue the initial x value.
     * @param yValue the initial y value.
     * @param zValue the initial z value.
     */
    DAEFloat3(float xValue, float yValue, float zValue){
        x = xValue;
        y = yValue;
        z = zValue;
    }

    /**
     * Creates a new DAEFloat3 object.
     * @param xValue the initial x value.
     * @param yValue the initial y value.
     * @param zValue the initial z value.
     * @param normalize if true the DAEFloat3 object will be normalized
     * immediatly, if false the x,y & z values will not be transformed.
     */
    DAEFloat3(float xValue, float yValue, float zValue, boolean normalize){
        this(xValue,yValue,zValue);
        if (normalize)
            Normalize();
    }

    /**
     * Sets the 3 values of the DAEFloat3 object in one go.
     * @param x the new x value.
     * @param y the new y value.
     * @param z the new z value.
     */
    void SetValues(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /**
     * Copies the values from another DAEFloat3 object.
     * @param toCopy the DAEFloat3 object to copy.
     */
    void SetValues(DAEFloat3 toCopy) {
        this.x = toCopy.x;
        this.y = toCopy.y;
        this.z = toCopy.z;
    }

    /**
     * Normalizes this DAEFloat3 object.
     * WARNING : this method changes the internal state.
     */
    void Normalize() {
        float length = (float) Math.sqrt(x * x + y * y + z * z);
        if (length > 0.0f) {
            x /= length;
            y /= length;
            z /= length;
        }
    }

    /**
     * Returns the length of this DAEFloat3 object.
     * @return the lenght of this DAEFloat3 object.
     */
    float Length() {
        return (float) Math.sqrt(x * x + y * y + z * z);
    }

    /**
     * The Squared length of this DAEFloat3 object.
     * Can be used to compare lengths of different DAEFloat3 objects,
     * in other when it is not needed to calculate the squared length.
     * @return the squared length of this DAEFloat3 object.
     */
    float SquaredLength() {
        return x * x + y * y + z * z;
    }

    /**
     * Implements the formula : result = this + position.
     * This method does not change the internal data.
     * @param position the vector to add to this vector.
     * @param outResult stores the result of the add operation.
     * @return the DAEFloat3 result object.
     */
    DAEFloat3 Add(DAEFloat3 outResult, DAEFloat3 position) {
        outResult.SetValues(x + position.x, y + position.y, z + position.z);
        return outResult;
    }

    /**
     * Implements the formula : result = this - vector.
     * This method does not change the internal data.
     * @param position the vector to subtract from this vector.
     * @param outResult stores the result of the subtract operation.
     * @return the DAEFloat3 result object.
     */
    DAEFloat3 Subtract(DAEFloat3 outResult, DAEFloat3 position) {
        outResult.SetValues(x - position.x, y - position.y, z - position.z);
        return outResult;
    }

    /**
     * Implements the formula :
     * 1) result = to - this
     * 2) normalize(result)
     * @param to the position the result vector will point at.
     * @param outResult stores the result of the create vector operation.
     * @return the normalized vector between to and this.
     */
    DAEFloat3 CreateVector(DAEFloat3 outResult, DAEFloat3 to) {
        float xr = to.x - x;
        float yr = to.y - y;
        float zr = to.z - z;
        outResult.SetValues(xr, yr, zr);
        outResult.Normalize();
        return outResult;
    }

    /**
     * Creates a vector that is perpendicular to this vector.
     * @param outResult the resulting vector.
     * @param normalizeResult set this to true if the result should be normalized.
     */
    void CreatePerpendicularVector(DAEFloat3 outResult) {
        DAEFloat3 frame = new DAEFloat3();
        float absx = Math.abs(x);
        float absy = Math.abs(y);
        float absz = Math.abs(z);
        if (absx < absy && absx < absz) {
            frame.SetValues(1, 0, 0);
        } else if (absy < absx && absy < absz) {
            frame.SetValues(0, 1, 0);
        } else {
            frame.SetValues(0, 0, 1);
        }

        CrossAndNormalize(outResult, frame);
    }

    /**
     * Implements the formula :
     * 1) result = cross(this,vector);
     * @param vector the vector to calculate the cross product for.
     * @param outResult stores the result of the cross operation.
     * @return the resulting cross product.
     */
    DAEFloat3 Cross(DAEFloat3 outResult, DAEFloat3 vector) {
        float xr = this.y * vector.z - this.z * vector.y;
        float yr = this.z * vector.x - this.x * vector.z;
        float zr = this.x * vector.y - this.y * vector.x;
        outResult.SetValues(xr, yr, zr);
        return outResult;
    }

    /**
     * Implements the formula :
     * 1) result = cross(this,vector);
     * 2) normalize(result)
     * @param vector the vector to calculate the cross product for.
     * @param outResult stores the result of the cross and normalize operation.
     * @return the resulting cross product.
     */
    DAEFloat3 CrossAndNormalize(DAEFloat3 outResult, DAEFloat3 vector) {
        float xr = this.y * vector.z - this.z * vector.y;
        float yr = this.z * vector.x - this.x * vector.z;
        float zr = this.x * vector.y - this.y * vector.x;
        outResult.SetValues(xr, yr, zr);
        outResult.Normalize();
        return outResult;
    }

    /**
     * Computes the dot product with another DAEFloat3 object.
     * @param vector2 the second operand in the dot product.
     * @return the dot product.
     */
    float Dot(DAEFloat3 vector2) {
        return this.x * vector2.x + this.y * vector2.y + this.z * vector2.z;
    }

    /**
     * Implements the formula : result = this + scale*vector
     * @param outResult stores the result of the Add operation.
     * @param scale the scale of the vector.
     * @param vector the vector to add to this vector.
     */
    DAEFloat3 Add(DAEFloat3 outResult, float scale, DAEFloat3 vector) {
        float scalex = scale * vector.x;
        float scaley = scale * vector.y;
        float scalez = scale * vector.z;

        float rx = x + scalex;
        float ry = y + scaley;
        float rz = z + scalez;

        outResult.SetValues(rx, ry, rz);
        return outResult;
    }

    /**
     * Scales this DAE3Float object and returns the result.
     * result = this*scale;
     * @param outResult stores the result of the scale operation.
     * @float scale the value of the multiplier
     */
    DAEFloat3 Scale(DAEFloat3 outResult, float scale) {
        outResult.SetValues(x * scale, y * scale, z * scale);
        return outResult;
    }

    /**
     * Clamps the values in this DAEFloat3 object to the specified
     * range.
     * @param min the minimum value for the x,y and z members.
     * @param max the maximum value for the x,y and z members.
     */
    void Clamp(float min, float max) {
        if (x > max) {
            x = max;
        } else if (x < min) {
            x = min;
        }

        if (y > max) {
            y = max;
        } else if (y < min) {
            y = min;
        }

        if (z > max) {
            z = max;
        } else if (z < min) {
            z = min;
        }
    }

    /**
     * Projects this vector onto the plane going through the origin and
     * defined by a normal vector.
     * @param normal the normal vector.
     * @return the result of the projection.
     */
    DAEFloat3 Project(DAEFloat3 normal) {
        float dot = normal.Dot(this);
        DAEFloat3 result = new DAEFloat3(normal.x, normal.y, normal.z);
        result.Scale(result, dot);
        result.Add(result, this);
        return result;
    }

    /**
     * Calculates the barycentric coordinates of the point p given the
     * triangle as defined by p1, p2 & p3.
     * @param p the point to convert to barycentric coordinates.
     * @param p1 the first vertex of the triangle.
     * @param p2 the second vertex of the triangle.
     * @param p3 the third vertex of the triangle.
     * @return the barycentric coordinates of the point p.
     */
    static DAEFloat3 BarycentricCoordinates(DAEFloat3 p, DAEFloat3 p1, DAEFloat3 p2, DAEFloat3 p3) {
        DAEFloat3 cross = new DAEFloat3();
        DAEFloat3 e13 = new DAEFloat3();
        DAEFloat3 e23 = new DAEFloat3();
        DAEFloat3 ep3 = new DAEFloat3();


        p1.Subtract(e13,p3);
        p2.Subtract(e23,p3);
        p.Subtract(ep3,p3);

        e13.Cross(cross, e23);

        float ax = Math.abs(cross.x);
        float ay = Math.abs(cross.y);
        float az = Math.abs(cross.z);

        DAEFloat3 result = new DAEFloat3();

        if (az > ax && az > ay) {
            // use xy plane.
            // l1
            result.x = (e23.y * ep3.x - e23.x * ep3.y) / (e23.y * e13.x - e23.x * e13.y);
            result.y = (-e13.y * ep3.x + e13.x * ep3.y) / (e23.y * e13.x - e23.x * e13.y);
            result.z = 1 - result.x - result.y;
        } else if (ay > ax && ay > az) {
            // use xz plane
            result.x = (e23.z * ep3.x - e23.x * ep3.z) / (e23.z * e13.x - e23.x * e13.z);
            result.y = (-e13.z * ep3.x + e13.x * ep3.z) / (e23.z * e13.x - e23.x * e13.z);
            result.z = 1 - result.x - result.y;
        } else {
            result.x = (e23.z * ep3.y - e23.y * ep3.z) / (e23.z * e13.y - e23.y * e13.z);
            result.y = (-e13.z * ep3.y + e13.y * ep3.z) / (e23.z * e13.y - e23.y * e13.z);
            result.z = 1 - result.y - result.x;
        }
        return result;
    }
    /**
     * The DAEFloat3 members
     */
    float x, y, z;
}
